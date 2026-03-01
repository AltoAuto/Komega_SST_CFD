from pathlib import Path
import time as time_module
import importlib

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bc
import turbulence
from mesh import make_mesh, ensure_single_mesh_file

try:
    from numba_kernels import advect_scalar_numba, grad_scalar_numba
    NUMBA_AVAILABLE = True
except Exception:
    advect_scalar_numba = None
    grad_scalar_numba = None
    NUMBA_AVAILABLE = False

USE_NUMBA = False


def _require_packages(use_numba=False):
    """Ensure required runtime packages are available before solving."""
    required = ["numpy", "scipy"]
    if use_numba:
        required.append("numba")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required packages: {missing_list}")

def _linear_face_weights(left, right, face):
    """Compute linear interpolation weights for a face between two cells."""
    d = right - left
    denom = d[..., 0] ** 2 + d[..., 1] ** 2
    denom_safe = np.where(denom > 1.0e-14, denom, 1.0)
    t = ((face - left) * d).sum(axis=-1) / denom_safe
    t = np.clip(t, 0.0, 1.0)
    w_right = t
    w_left = 1.0 - t
    return w_left, w_right

def _face_weights_i(mesh):
    """Compute interpolation weights for i-direction faces."""
    if "w_left_i" in mesh and "w_right_i" in mesh:
        return mesh["w_left_i"], mesh["w_right_i"]
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    w_left = np.full((count_i + 1, count_j), 0.5, dtype=float)
    w_right = np.full((count_i + 1, count_j), 0.5, dtype=float)

    cell_center = mesh["cell_center"]
    face_center = mesh["face_center"]

    left = cell_center[:-1, :, :]
    right = cell_center[1:, :, :]
    face = face_center[1:, :, 0, :]
    wl, wr = _linear_face_weights(left, right, face)
    w_left[1:count_i, :] = wl
    w_right[1:count_i, :] = wr
    return w_left, w_right

def _face_weights_j(mesh):
    """Compute interpolation weights for j-direction faces."""
    if "w_south_j" in mesh and "w_north_j" in mesh:
        return mesh["w_south_j"], mesh["w_north_j"]
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    w_south = np.full((count_i, count_j + 1), 0.5, dtype=float)
    w_north = np.full((count_i, count_j + 1), 0.5, dtype=float)

    cell_center = mesh["cell_center"]
    face_center = mesh["face_center"]

    south = cell_center[:, :-1, :]
    north = cell_center[:, 1:, :]
    face = face_center[:, 1:, 1, :]
    ws, wn = _linear_face_weights(south, north, face)
    w_south[:, 1:count_j] = ws
    w_north[:, 1:count_j] = wn
    return w_south, w_north

def _face_normals(mesh):
    """Return face normals for internal and boundary faces."""
    if "face_normal_i" in mesh and "face_normal_j" in mesh:
        return mesh["face_normal_i"], mesh["face_normal_j"]
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    face_normal_i = np.zeros((count_i + 1, count_j, 2), dtype=float)
    face_normal_j = np.zeros((count_i, count_j + 1, 2), dtype=float)

    face_normal_i[:count_i, :, :] = mesh["face_normal"][:, :, 0, :]
    face_normal_j[:, :count_j, :] = mesh["face_normal"][:, :, 1, :]

    node = mesh["node"]

    n0 = node[count_i, 0:count_j, :]
    n1 = node[count_i, 1 : count_j + 1, :]
    t = n1 - n0
    n = np.stack((t[..., 1], -t[..., 0]), axis=-1)
    face_center = 0.5 * (n0 + n1)
    cell_center = mesh["cell_center"][count_i - 1, :, :]
    s = face_center - cell_center
    dot = np.sum(n * s, axis=-1)
    n *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
    face_normal_i[count_i, :, :] = n

    n0 = node[0:count_i, count_j, :]
    n1 = node[1 : count_i + 1, count_j, :]
    t = n1 - n0
    n = np.stack((-t[..., 1], t[..., 0]), axis=-1)
    face_center = 0.5 * (n0 + n1)
    cell_center = mesh["cell_center"][:, count_j - 1, :]
    s = face_center - cell_center
    dot = np.sum(n * s, axis=-1)
    n *= np.where(dot < 0.0, -1.0, 1.0)[..., None]
    face_normal_j[:, count_j, :] = n

    return face_normal_i, face_normal_j

def _face_centers(mesh):
    """Return face centers for internal and boundary faces."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    face_center_i = np.zeros((count_i + 1, count_j, 2), dtype=float)
    face_center_j = np.zeros((count_i, count_j + 1, 2), dtype=float)

    face_center_i[:count_i, :, :] = mesh["face_center"][:, :, 0, :]
    face_center_j[:, :count_j, :] = mesh["face_center"][:, :, 1, :]

    node = mesh["node"]
    n0 = node[count_i, 0:count_j, :]
    n1 = node[count_i, 1 : count_j + 1, :]
    face_center_i[count_i, :, :] = 0.5 * (n0 + n1)

    n0 = node[0:count_i, count_j, :]
    n1 = node[1 : count_i + 1, count_j, :]
    face_center_j[:, count_j, :] = 0.5 * (n0 + n1)

    return face_center_i, face_center_j


def _diffusion_coefficients(mesh, nu):
    """Compute diffusion coefficients for faces (supports variable nu)."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    cell_center = mesh["cell_center"]

    face_normal_i, face_normal_j = _face_normals(mesh)
    s_i = np.hypot(face_normal_i[..., 0], face_normal_i[..., 1])
    s_j = np.hypot(face_normal_j[..., 0], face_normal_j[..., 1])

    if "diff_d_i" in mesh and "diff_d_j" in mesh:
        d_i = mesh["diff_d_i"]
        d_j = mesh["diff_d_j"]
    else:
        face_center_i, face_center_j = _face_centers(mesh)
        n_i = face_normal_i / s_i[..., None]
        n_j = face_normal_j / s_j[..., None]

        d_i = np.zeros((count_i + 1, count_j), dtype=float)
        d_j = np.zeros((count_i, count_j + 1), dtype=float)

        delta_i = cell_center[1:, :, :] - cell_center[:-1, :, :]
        d_i[1:count_i, :] = np.abs(np.sum(delta_i * n_i[1:count_i, :, :], axis=-1))

        delta_j = cell_center[:, 1:, :] - cell_center[:, :-1, :]
        d_j[:, 1:count_j] = np.abs(np.sum(delta_j * n_j[:, 1:count_j, :], axis=-1))

        s_w = np.abs(
            np.sum((cell_center[0, :, :] - face_center_i[0, :, :]) * n_i[0, :, :], axis=-1)
        )
        d_i[0, :] = 2.0 * s_w

        s_e = np.abs(
            np.sum(
                (cell_center[count_i - 1, :, :] - face_center_i[count_i, :, :])
                * n_i[count_i, :, :],
                axis=-1,
            )
        )
        d_i[count_i, :] = 2.0 * s_e

        s_s = np.abs(
            np.sum((cell_center[:, 0, :] - face_center_j[:, 0, :]) * n_j[:, 0, :], axis=-1)
        )
        d_j[:, 0] = 2.0 * s_s

        s_n = np.abs(
            np.sum(
                (cell_center[:, count_j - 1, :] - face_center_j[:, count_j, :])
                * n_j[:, count_j, :],
                axis=-1,
            )
        )
        d_j[:, count_j] = 2.0 * s_n

        d_i = np.where(d_i > 1.0e-14, d_i, 1.0e-14)
        d_j = np.where(d_j > 1.0e-14, d_j, 1.0e-14)

    if np.isscalar(nu):
        nu_i = nu
        nu_j = nu
    else:
        w_left, w_right = _face_weights_i(mesh)
        w_south, w_north = _face_weights_j(mesh)
        nu_i = np.zeros((count_i + 1, count_j), dtype=float)
        nu_j = np.zeros((count_i, count_j + 1), dtype=float)
        nu_left = nu[:-1, :]
        nu_right = nu[1:, :]
        nu_i[1:count_i, :] = (
            w_left[1:count_i, :] * nu_left + w_right[1:count_i, :] * nu_right
        )
        nu_i[0, :] = nu[0, :]
        nu_i[count_i, :] = nu[-1, :]
        nu_south = nu[:, :-1]
        nu_north = nu[:, 1:]
        nu_j[:, 1:count_j] = (
            w_south[:, 1:count_j] * nu_south + w_north[:, 1:count_j] * nu_north
        )
        nu_j[:, 0] = nu[:, 0]
        nu_j[:, count_j] = nu[:, -1]

    a_i = nu_i * s_i / d_i
    a_j = nu_j * s_j / d_j
    return a_i, a_j


def _parse_index_range(index_range, max_index):
    """Normalize and validate an index range on a boundary."""
    if index_range is None:
        return 0, max_index
    if len(index_range) != 2:
        raise ValueError("index_range must be (start, end)")
    start, end = index_range
    if start < 0 or end < start or end > max_index:
        raise ValueError("index_range out of bounds")
    return start, end


def _pressure_dirichlet_masks(bc_cfg, count_i, count_j):
    """Build masks for Dirichlet pressure boundaries."""
    masks = {
        "i_min": np.zeros(count_j, dtype=bool),
        "i_max": np.zeros(count_j, dtype=bool),
        "j_min": np.zeros(count_i, dtype=bool),
        "j_max": np.zeros(count_i, dtype=bool),
    }
    has_dirichlet = False

    for boundary, segments in bc_cfg.items():
        if boundary == "pressure_reference":
            continue
        if segments is None:
            continue
        if isinstance(segments, dict):
            segments = [segments]
        for seg in segments:
            if seg.get("type") != "outlet" and "pressure" not in seg:
                continue
            index_range = seg.get("range")
            if boundary in ("i_min", "i_max"):
                start, end = _parse_index_range(index_range, count_j - 1)
                masks[boundary][start : end + 1] = True
            elif boundary in ("j_min", "j_max"):
                start, end = _parse_index_range(index_range, count_i - 1)
                masks[boundary][start : end + 1] = True
            else:
                raise ValueError(f"Unknown boundary: {boundary}")
            has_dirichlet = True

    return masks, has_dirichlet


def _apply_pressure_correction_bc(p_corr, bc_cfg, count_i, count_j):
    """Apply pressure-correction BCs for periodic and wall boundaries."""
    range_i, range_j = bc.get_periodic_pairs(bc_cfg, count_i, count_j)
    masks, _ = _pressure_dirichlet_masks(bc_cfg, count_i, count_j)

    periodic_i = range_i is not None and not (
        np.any(masks["i_min"]) or np.any(masks["i_max"])
    )
    periodic_j = range_j is not None and not (
        np.any(masks["j_min"]) or np.any(masks["j_max"])
    )

    if periodic_i:
        start, end = range_i
        j_slice = slice(start + 1, end + 2)
        p_corr[0, j_slice] = p_corr[count_i, j_slice]
        p_corr[count_i + 1, j_slice] = p_corr[1, j_slice]

    if periodic_j:
        start, end = range_j
        i_slice = slice(start + 1, end + 2)
        p_corr[i_slice, 0] = p_corr[i_slice, count_j]
        p_corr[i_slice, count_j + 1] = p_corr[i_slice, 1]

    if not periodic_i:
        phi_p = p_corr[1, 1:-1]
        mask = masks["i_min"]
        p_corr[0, 1:-1] = np.where(mask, -phi_p, phi_p)

        phi_p = p_corr[count_i, 1:-1]
        mask = masks["i_max"]
        p_corr[count_i + 1, 1:-1] = np.where(mask, -phi_p, phi_p)

    if not periodic_j:
        phi_p = p_corr[1:-1, 1]
        mask = masks["j_min"]
        p_corr[1:-1, 0] = np.where(mask, -phi_p, phi_p)

        phi_p = p_corr[1:-1, count_j]
        mask = masks["j_max"]
        p_corr[1:-1, count_j + 1] = np.where(mask, -phi_p, phi_p)


def grad_scalar(phi, mesh):
    """Compute cell-centered gradient of a scalar field."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    volume = mesh["cell_volume"]

    w_left, w_right = _face_weights_i(mesh)
    w_south, w_north = _face_weights_j(mesh)
    face_normal_i, face_normal_j = _face_normals(mesh)

    if USE_NUMBA and NUMBA_AVAILABLE:
        return grad_scalar_numba(
            phi,
            w_left,
            w_right,
            w_south,
            w_north,
            face_normal_i,
            face_normal_j,
            volume,
        )

    phi_left = phi[0:-1, 1:-1]
    phi_right = phi[1:, 1:-1]
    phi_i = w_left * phi_left + w_right * phi_right

    phi_south = phi[1:-1, 0:-1]
    phi_north = phi[1:-1, 1:]
    phi_j = w_south * phi_south + w_north * phi_north

    flux = (
        face_normal_i[1:, :, :] * phi_i[1:, :, None]
        - face_normal_i[:-1, :, :] * phi_i[:-1, :, None]
        + face_normal_j[:, 1:, :] * phi_j[:, 1:, None]
        - face_normal_j[:, :-1, :] * phi_j[:, :-1, None]
    )
    return flux / volume[:, :, None]


def load_mesh(mesh_cfg):
    """Load mesh from disk, generating it if needed."""
    mesh_path = Path(mesh_cfg["path"])
    ensure_single_mesh_file(mesh_path)
    if not mesh_path.exists():
        make_mesh(mesh_cfg)
    return {key: value for key, value in np.load(mesh_path).items()}


def initialize_fields(mesh, initial_cfg):
    """Initialize flow and scalar fields with ghost cells."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    shape = (count_i + 2, count_j + 2)
    interior = (slice(1, -1), slice(1, -1))
    fields = {
        "u": np.zeros(shape, dtype=float),
        "v": np.zeros(shape, dtype=float),
        "p": np.zeros(shape, dtype=float),
    }
    fields["u"][interior] = initial_cfg["u"]
    fields["v"][interior] = initial_cfg["v"]
    fields["p"][interior] = initial_cfg["p"]
    for name, value in initial_cfg.get("scalars", {}).items():
        fields[name] = np.zeros(shape, dtype=float)
        fields[name][interior] = value
    return fields


def _load_field_array(target, arr):
    """Load field array into target, supporting cell-only inputs."""
    if arr.shape == target.shape:
        target[:, :] = arr
        return
    if arr.shape == target[1:-1, 1:-1].shape:
        target[1:-1, 1:-1] = arr
        return
    raise ValueError(f"Field shape {arr.shape} does not match target {target.shape}.")


def load_state(state_path, mesh, initial_cfg):
    """Load a saved state and map into solver field arrays."""
    data = np.load(state_path)
    fields = initialize_fields(mesh, initial_cfg)
    meta_keys = {"step", "time", "cell_center"}

    for key in data.files:
        if key in meta_keys:
            continue
        arr = data[key]
        if key in fields:
            _load_field_array(fields[key], arr)
        else:
            fields[key] = np.zeros_like(fields["p"])
            _load_field_array(fields[key], arr)

    step = int(data["step"]) if "step" in data else None
    time = float(data["time"]) if "time" in data else None
    return fields, step, time


def apply_boundary_conditions(fields, mesh, bc_cfg):
    """Apply flow and turbulence boundary conditions."""
    bc.apply_boundary_conditions(fields, mesh, bc_cfg)


def _pressure_effective(fields, turb_cfg):
    """Return pressure including turbulence isotropic term if enabled."""
    if (
        turb_cfg
        and turb_cfg.get("enabled", False)
        and turb_cfg.get("model") not in (None, "none")
        and "k" in fields
    ):
        return fields["p"] + (2.0 / 3.0) * fields["k"]
    return fields["p"]


def _momentum_diagonal(mesh, nu, dt):
    """Assemble diagonal coefficients for momentum equations."""
    volume = mesh["cell_volume"]
    a_i, a_j = _diffusion_coefficients(mesh, nu)
    dt_field = np.asarray(dt)
    if dt_field.ndim == 0:
        inv_dt = 1.0 / float(dt_field)
        a_p = volume * inv_dt
    else:
        if dt_field.shape != volume.shape:
            raise ValueError("dt array must match cell_volume shape.")
        a_p = volume / dt_field
    a_p = a_p + a_i[:-1, :] + a_i[1:, :] + a_j[:, :-1] + a_j[:, 1:]
    return a_p


def _face_mass_flux(fields, mesh, a_p, use_rhie_chow=True, p_field=None):
    """Compute face mass fluxes with optional Rhie-Chow correction."""
    u = fields["u"]
    v = fields["v"]
    p = p_field if p_field is not None else fields["p"]

    w_left, w_right = _face_weights_i(mesh)
    w_south, w_north = _face_weights_j(mesh)
    face_normal_i, face_normal_j = _face_normals(mesh)


    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    a_f_i = np.zeros((count_i + 1, count_j), dtype=float)
    a_f_j = np.zeros((count_i, count_j + 1), dtype=float)
    a_f_i[1:count_i, :] = (
        w_left[1:count_i, :] * a_p[:-1, :] + w_right[1:count_i, :] * a_p[1:, :]
    )
    a_f_j[:, 1:count_j] = (
        w_south[:, 1:count_j] * a_p[:, :-1] + w_north[:, 1:count_j] * a_p[:, 1:]
    )
    a_f_i[0, :] = a_p[0, :]
    a_f_i[count_i, :] = a_p[-1, :]
    a_f_j[:, 0] = a_p[:, 0]
    a_f_j[:, count_j] = a_p[:, -1]

    a_f_i = np.where(a_f_i > 1.0e-14, a_f_i, 1.0e-14)
    a_f_j = np.where(a_f_j > 1.0e-14, a_f_j, 1.0e-14)

    s_i = np.hypot(face_normal_i[..., 0], face_normal_i[..., 1])
    s_j = np.hypot(face_normal_j[..., 0], face_normal_j[..., 1])

    u_left = u[0:-1, 1:-1]
    u_right = u[1:, 1:-1]
    v_left = v[0:-1, 1:-1]
    v_right = v[1:, 1:-1]

    u_i = w_left * u_left + w_right * u_right
    v_i = w_left * v_left + w_right * v_right

    u_south = u[1:-1, 0:-1]
    u_north = u[1:-1, 1:]
    v_south = v[1:-1, 0:-1]
    v_north = v[1:-1, 1:]

    u_j = w_south * u_south + w_north * u_north
    v_j = w_south * v_south + w_north * v_north

    flux_i = u_i * face_normal_i[..., 0] + v_i * face_normal_i[..., 1]
    flux_j = u_j * face_normal_j[..., 0] + v_j * face_normal_j[..., 1]

    if use_rhie_chow:
        dp_i = p[1:, 1:-1] - p[0:-1, 1:-1]
        dp_j = p[1:-1, 1:] - p[1:-1, 0:-1]

        flux_i -= (s_i ** 2) / a_f_i * dp_i
        flux_j -= (s_j ** 2) / a_f_j * dp_j
    return flux_i, flux_j


def _face_volumetric_flux(fields, mesh):
    """Compute face volumetric fluxes without pressure correction."""
    u = fields["u"]
    v = fields["v"]

    w_left, w_right = _face_weights_i(mesh)
    w_south, w_north = _face_weights_j(mesh)
    face_normal_i, face_normal_j = _face_normals(mesh)


    u_left = u[0:-1, 1:-1]
    u_right = u[1:, 1:-1]
    v_left = v[0:-1, 1:-1]
    v_right = v[1:, 1:-1]

    u_i = w_left * u_left + w_right * u_right
    v_i = w_left * v_left + w_right * v_right

    u_south = u[1:-1, 0:-1]
    u_north = u[1:-1, 1:]
    v_south = v[1:-1, 0:-1]
    v_north = v[1:-1, 1:]

    u_j = w_south * u_south + w_north * u_north
    v_j = w_south * v_south + w_north * v_north

    flux_i = u_i * face_normal_i[..., 0] + v_i * face_normal_i[..., 1]
    flux_j = u_j * face_normal_j[..., 0] + v_j * face_normal_j[..., 1]
    return flux_i, flux_j


def compute_adaptive_dt(fields, mesh, cfl, dt_max, dt_min=0.0, cfl_diff=None, nu_eff=None):
    """Compute a global time step from advective and diffusive CFL constraints."""
    volume = mesh["cell_volume"]
    dt_adv = float("inf")
    if cfl is not None:
        flux_i, flux_j = _face_volumetric_flux(fields, mesh)
        flux_sum = (
            np.abs(flux_i[:-1, :])
            + np.abs(flux_i[1:, :])
            + np.abs(flux_j[:, :-1])
            + np.abs(flux_j[:, 1:])
        )
        small = 1.0e-14
        safe = np.where(flux_sum > small, flux_sum, small)
        dt_cell = cfl * volume / safe
        dt_adv = float(np.min(dt_cell))

    dt_diff = float("inf")
    if cfl_diff is not None:
        if nu_eff is None:
            nu_eff = 0.0
        spacing_i = mesh["cell_spacing_i"]
        spacing_j = mesh["cell_spacing_j"]
        length = np.minimum(spacing_i, spacing_j)
        nu_safe = np.where(nu_eff > 0.0, nu_eff, 1.0e-14)
        dt_cell = cfl_diff * (length ** 2) / nu_safe
        dt_diff = float(np.min(dt_cell))

    dt = min(dt_adv, dt_diff)
    if not np.isfinite(dt):
        dt = dt_max
    if dt_min is not None:
        dt = max(dt, float(dt_min))
    if dt_max is not None:
        dt = min(dt, float(dt_max))
    return dt


def _dt_limit_estimate(fields, mesh, cfl, cfl_diff, nu_eff):
    """Estimate minimum advective/diffusive dt for reporting."""
    volume = mesh["cell_volume"]
    dt_adv_min = float("inf")
    if cfl is not None:
        flux_i, flux_j = _face_volumetric_flux(fields, mesh)
        flux_sum = (
            np.abs(flux_i[:-1, :])
            + np.abs(flux_i[1:, :])
            + np.abs(flux_j[:, :-1])
            + np.abs(flux_j[:, 1:])
        )
        small = 1.0e-14
        safe = np.where(flux_sum > small, flux_sum, small)
        dt_adv_min = float(np.min(cfl * volume / safe))

    dt_diff_min = float("inf")
    if cfl_diff is not None:
        if nu_eff is None:
            nu_eff = 0.0
        spacing_i = mesh["cell_spacing_i"]
        spacing_j = mesh["cell_spacing_j"]
        length = np.minimum(spacing_i, spacing_j)
        nu_safe = np.where(nu_eff > 0.0, nu_eff, 1.0e-14)
        dt_diff_min = float(np.min(cfl_diff * (length ** 2) / nu_safe))
    return dt_adv_min, dt_diff_min


def _parse_value_schedule(schedule, name):
    """Normalize and validate a (value, step) schedule list."""
    if schedule is None:
        return None
    items = []
    for entry in schedule:
        if len(entry) != 2:
            raise ValueError(f"{name} entries must be (value, step).")
        value, step = entry
        items.append((float(value), int(step)))
    items.sort(key=lambda x: x[1])
    steps = [step for _, step in items]
    if any(step < 0 for step in steps):
        raise ValueError(f"{name} steps must be non-negative.")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f"{name} steps must be strictly increasing.")
    return items


def _scheduled_value(step, base_value, schedule):
    """Return a scheduled value for the given step."""
    if schedule is None:
        return base_value
    if not schedule:
        return base_value
    if step <= schedule[0][1]:
        return schedule[0][0]
    for (val_a, step_a), (val_b, step_b) in zip(schedule[:-1], schedule[1:]):
        if step <= step_b:
            if step_b == step_a:
                return val_b
            t = (step - step_a) / float(step_b - step_a)
            return (1.0 - t) * val_a + t * val_b
    return schedule[-1][0]


def advect_scalar(phi, mesh, flux_i, flux_j):
    """Compute first-order upwind advection for a scalar."""
    volume = mesh["cell_volume"]
    if USE_NUMBA and NUMBA_AVAILABLE:
        return advect_scalar_numba(phi, flux_i, flux_j, volume)

    phi_left = phi[0:-1, 1:-1]
    phi_right = phi[1:, 1:-1]
    phi_south = phi[1:-1, 0:-1]
    phi_north = phi[1:-1, 1:]

    phi_i = np.where(flux_i >= 0.0, phi_left, phi_right)
    phi_j = np.where(flux_j >= 0.0, phi_south, phi_north)

    div = (
        flux_i[1:, :] * phi_i[1:, :]
        - flux_i[:-1, :] * phi_i[:-1, :]
        + flux_j[:, 1:] * phi_j[:, 1:]
        - flux_j[:, :-1] * phi_j[:, :-1]
    )
    return -div / volume


def compute_advective_rhs(fields, mesh, nu, dt, use_rhie_chow=True, p_field=None):
    """Compute advective RHS for all transported fields."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    a_p = _momentum_diagonal(mesh, nu, dt)
    flux_i, flux_j = _face_mass_flux(
        fields, mesh, a_p, use_rhie_chow=use_rhie_chow, p_field=p_field
    )

    rhs = {}
    for name in ("u", "v"):
        phi = fields[name]
        rhs_field = np.zeros((count_i + 2, count_j + 2), dtype=float)
        rhs_field[1:-1, 1:-1] = advect_scalar(phi, mesh, flux_i, flux_j)
        rhs[name] = rhs_field

    rhs["flux_i"] = flux_i
    rhs["flux_j"] = flux_j
    return rhs


def _assemble_implicit_system(phi, mesh, nu, dt, source, bc_cfg=None, diag_extra=None):
    """Assemble the implicit diffusion/transient system for a scalar."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    volume = mesh["cell_volume"]
    dt_field = np.asarray(dt)
    if dt_field.ndim == 0:
        inv_dt = 1.0 / float(dt_field)
        dt_scalar = True
    else:
        if dt_field.shape != volume.shape:
            raise ValueError("dt array must match cell_volume shape.")
        inv_dt = 1.0 / dt_field
        dt_scalar = False

    a_i, a_j = _diffusion_coefficients(mesh, nu)
    periodic_i = False
    periodic_j = False
    if bc_cfg is not None:
        range_i, range_j = bc.get_periodic_pairs(bc_cfg, count_i, count_j)
        periodic_i = range_i is not None
        periodic_j = range_j is not None

    a_w = a_i[:-1, :]
    a_e = a_i[1:, :]
    a_s = a_j[:, :-1]
    a_n = a_j[:, 1:]

    inv_dt_cell = inv_dt if dt_scalar else inv_dt
    b = volume * inv_dt_cell * phi[1:-1, 1:-1] + volume * source

    if not periodic_i:
        b[0, :] += a_w[0, :] * phi[0, 1:-1]
        b[-1, :] += a_e[-1, :] * phi[count_i + 1, 1:-1]
    if not periodic_j:
        b[:, 0] += a_s[:, 0] * phi[1:-1, 0]
        b[:, -1] += a_n[:, -1] * phi[1:-1, count_j + 1]

    a_p = volume * inv_dt_cell + a_w + a_e + a_s + a_n
    if diag_extra is not None:
        a_p += volume * diag_extra

    w_off = a_w.copy()
    w_off[0, :] = 0.0
    e_off = a_e.copy()
    e_off[-1, :] = 0.0
    s_off = a_s.copy()
    s_off[:, 0] = 0.0
    n_off = a_n.copy()
    n_off[:, -1] = 0.0

    n = count_i * count_j
    diag = a_p.ravel()
    diags = [
        diag,
        -w_off.ravel()[count_j:],
        -e_off.ravel()[:-count_j],
        -s_off.ravel()[1:],
        -n_off.ravel()[:-1],
    ]
    offsets = [0, -count_j, count_j, -1, 1]
    a_mat = sp.diags(diags, offsets, shape=(n, n), format="csr")

    rows = []
    cols = []
    data = []
    if periodic_i:
        j_idx = np.arange(count_j)
        rows.append(j_idx)
        cols.append((count_i - 1) * count_j + j_idx)
        data.append(-a_w[0, :])
        rows.append((count_i - 1) * count_j + j_idx)
        cols.append(j_idx)
        data.append(-a_e[-1, :])
    if periodic_j:
        i_idx = np.arange(count_i)
        rows.append(i_idx * count_j)
        cols.append(i_idx * count_j + (count_j - 1))
        data.append(-a_s[:, 0])
        rows.append(i_idx * count_j + (count_j - 1))
        cols.append(i_idx * count_j)
        data.append(-a_n[:, -1])
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        a_mat = a_mat + sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    return a_mat, b.ravel(), a_p


def _solve_linear_system(
    a_mat,
    b,
    tol=1.0e-8,
    maxiter=200,
    use_amg=False,
):
    """Solve a sparse linear system with optional AMG."""
    if use_amg:
        try:
            import pyamg
            ml = pyamg.ruge_stuben_solver(a_mat)
            x = ml.solve(b, tol=tol, maxiter=maxiter)
            return x
        except Exception as exc:
            raise RuntimeError(f"AMG solver failed: {exc}") from exc

    precond = None
    try:
        ilu = spla.spilu(a_mat.tocsc())
        precond = spla.LinearOperator(a_mat.shape, ilu.solve)
    except RuntimeError:
        precond = None

    x, info = spla.bicgstab(a_mat, b, rtol=tol, maxiter=maxiter, M=precond)
    if info == 0:
        return x

    x, info = spla.cg(a_mat, b, rtol=tol, maxiter=maxiter, M=precond)
    if info == 0:
        return x

    x = spla.spsolve(a_mat, b)
    return x


def _solve_pressure_system(a_mat, b, tol=1.0e-8, maxiter=200, amg_cache=None):
    """Solve the pressure Poisson system with AMG or CG."""
    try:
        import pyamg
        if amg_cache is None:
            amg_cache = {}
        ml = amg_cache.get("ml")
        if ml is None or amg_cache.get("shape") != a_mat.shape:
            ml = pyamg.ruge_stuben_solver(a_mat)
            amg_cache["ml"] = ml
            amg_cache["shape"] = a_mat.shape
        x = ml.solve(b, tol=tol, maxiter=maxiter)
        return x
    except ImportError:
        precond = None
        try:
            ilu = spla.spilu(a_mat.tocsc())
            precond = spla.LinearOperator(a_mat.shape, ilu.solve)
        except RuntimeError:
            precond = None

        x, info = spla.cg(a_mat, b, rtol=tol, maxiter=maxiter, M=precond)
        if info != 0:
            raise RuntimeError(f"Pressure solver did not converge (info={info}).")
        return x


def solve_momentum(
    fields,
    mesh,
    dt,
    nu,
    adv_rhs,
    bc_cfg,
    solver_cfg=None,
    p_field=None,
    pressure_gradient=(0.0, 0.0),
    extra_pressure_field=None,
):
    """Solve momentum predictor equations for u and v."""
    if solver_cfg is None:
        solver_cfg = {}
    tol = solver_cfg.get("tol", 1.0e-8)
    maxiter = solver_cfg.get("maxiter", 200)
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    if p_field is None:
        p_field = fields["p"]
    grad_p = grad_scalar(p_field, mesh)
    if extra_pressure_field is not None:
        grad_p = grad_p + grad_scalar(extra_pressure_field, mesh)
    force_u = -float(pressure_gradient[0])
    force_v = -float(pressure_gradient[1])
    source_u = adv_rhs["u"][1:-1, 1:-1] - grad_p[:, :, 0] + force_u
    source_v = adv_rhs["v"][1:-1, 1:-1] - grad_p[:, :, 1] + force_v

    a_u, b_u, diag_u = _assemble_implicit_system(
        fields["u"], mesh, nu, dt, source_u, bc_cfg=bc_cfg
    )
    a_v, b_v, diag_v = _assemble_implicit_system(
        fields["v"], mesh, nu, dt, source_v, bc_cfg=bc_cfg
    )

    u_sol = _solve_linear_system(a_u, b_u, tol=tol, maxiter=maxiter)
    v_sol = _solve_linear_system(a_v, b_v, tol=tol, maxiter=maxiter)

    res_u = a_u.dot(u_sol) - b_u
    res_v = a_v.dot(v_sol) - b_v
    mom_u_l2 = float(np.sqrt(np.mean(res_u ** 2)))
    mom_v_l2 = float(np.sqrt(np.mean(res_v ** 2)))

    fields["u"][1:-1, 1:-1] = u_sol.reshape((count_i, count_j))
    fields["v"][1:-1, 1:-1] = v_sol.reshape((count_i, count_j))

    return {
        "a_p_u": diag_u,
        "a_p_v": diag_v,
        "mom_u_l2": mom_u_l2,
        "mom_v_l2": mom_v_l2,
    }


def solve_pressure(
    fields,
    mesh,
    dt,
    nu,
    bc_cfg,
    solver_cfg=None,
    use_rhie_chow=True,
    p_field=None,
    a_p_override=None,
    amg_cache=None,
):
    """Solve pressure correction Poisson equation."""
    if solver_cfg is None:
        solver_cfg = {}
    tol = solver_cfg.get("tol", 1.0e-8)
    maxiter = solver_cfg.get("maxiter", 200)

    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    volume = mesh["cell_volume"]

    a_p = a_p_override if a_p_override is not None else _momentum_diagonal(mesh, nu, dt)
    flux_i, flux_j = _face_mass_flux(
        fields, mesh, a_p, use_rhie_chow=use_rhie_chow, p_field=p_field
    )
    div = (
        flux_i[1:, :] - flux_i[:-1, :] + flux_j[:, 1:] - flux_j[:, :-1]
    ) / volume
    rhs = -div

    w_left, w_right = _face_weights_i(mesh)
    w_south, w_north = _face_weights_j(mesh)
    face_normal_i, face_normal_j = _face_normals(mesh)

    a_f_i = np.zeros((count_i + 1, count_j), dtype=float)
    a_f_j = np.zeros((count_i, count_j + 1), dtype=float)
    a_f_i[1:count_i, :] = (
        w_left[1:count_i, :] * a_p[:-1, :] + w_right[1:count_i, :] * a_p[1:, :]
    )
    a_f_j[:, 1:count_j] = (
        w_south[:, 1:count_j] * a_p[:, :-1] + w_north[:, 1:count_j] * a_p[:, 1:]
    )
    a_f_i[0, :] = a_p[0, :]
    a_f_i[count_i, :] = a_p[-1, :]
    a_f_j[:, 0] = a_p[:, 0]
    a_f_j[:, count_j] = a_p[:, -1]

    a_f_i = np.where(a_f_i > 1.0e-14, a_f_i, 1.0e-14)
    a_f_j = np.where(a_f_j > 1.0e-14, a_f_j, 1.0e-14)

    s_i = np.hypot(face_normal_i[..., 0], face_normal_i[..., 1])
    s_j = np.hypot(face_normal_j[..., 0], face_normal_j[..., 1])
    d_f_i = (s_i ** 2) / a_f_i
    d_f_j = (s_j ** 2) / a_f_j
    masks, has_dirichlet = _pressure_dirichlet_masks(bc_cfg, count_i, count_j)
    range_i, range_j = bc.get_periodic_pairs(bc_cfg, count_i, count_j)
    periodic_i = range_i is not None and not (
        np.any(masks["i_min"]) or np.any(masks["i_max"])
    )
    periodic_j = range_j is not None and not (
        np.any(masks["j_min"]) or np.any(masks["j_max"])
    )

    a_w = d_f_i[:-1, :]
    a_e = d_f_i[1:, :]
    a_s = d_f_j[:, :-1]
    a_n = d_f_j[:, 1:]

    add_w = a_w.copy()
    add_e = a_e.copy()
    add_s = a_s.copy()
    add_n = a_n.copy()
    if not periodic_i:
        add_w[0, :] = np.where(masks["i_min"], 2.0 * a_w[0, :], 0.0)
        add_e[-1, :] = np.where(masks["i_max"], 2.0 * a_e[-1, :], 0.0)
    if not periodic_j:
        add_s[:, 0] = np.where(masks["j_min"], 2.0 * a_s[:, 0], 0.0)
        add_n[:, -1] = np.where(masks["j_max"], 2.0 * a_n[:, -1], 0.0)

    a_p_cell = add_w + add_e + add_s + add_n
    b = (volume * rhs).ravel()

    w_off = a_w.copy()
    w_off[0, :] = 0.0
    e_off = a_e.copy()
    e_off[-1, :] = 0.0
    s_off = a_s.copy()
    s_off[:, 0] = 0.0
    n_off = a_n.copy()
    n_off[:, -1] = 0.0

    n = count_i * count_j
    diag = a_p_cell.ravel()
    diags = [
        diag,
        -w_off.ravel()[count_j:],
        -e_off.ravel()[:-count_j],
        -s_off.ravel()[1:],
        -n_off.ravel()[:-1],
    ]
    offsets = [0, -count_j, count_j, -1, 1]
    a_mat = sp.diags(diags, offsets, shape=(n, n), format="csr")

    rows = []
    cols = []
    data = []
    if periodic_i:
        j_idx = np.arange(count_j)
        rows.append(j_idx)
        cols.append((count_i - 1) * count_j + j_idx)
        data.append(-a_w[0, :])
        rows.append((count_i - 1) * count_j + j_idx)
        cols.append(j_idx)
        data.append(-a_e[-1, :])
    if periodic_j:
        i_idx = np.arange(count_i)
        rows.append(i_idx * count_j)
        cols.append(i_idx * count_j + (count_j - 1))
        data.append(-a_s[:, 0])
        rows.append(i_idx * count_j + (count_j - 1))
        cols.append(i_idx * count_j)
        data.append(-a_n[:, -1])
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        a_mat = a_mat + sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    if not has_dirichlet:
        ref_cfg = bc_cfg.get("pressure_reference", {})
        ref_cell = ref_cfg.get("cell", (0, 0))
        ref_value = ref_cfg.get("value", 0.0)
        i_ref, j_ref = ref_cell
        row = i_ref * count_j + j_ref

        a_mat = a_mat.tolil()
        a_mat[row, :] = 0.0
        a_mat[row, row] = 1.0
        b[row] = ref_value
        a_mat = a_mat.tocsr()

    p_corr = _solve_pressure_system(
        a_mat, b, tol=tol, maxiter=maxiter, amg_cache=amg_cache
    )

    if "p_corr" not in fields:
        fields["p_corr"] = np.zeros((count_i + 2, count_j + 2), dtype=float)
    fields["p_corr"][1:-1, 1:-1] = p_corr.reshape((count_i, count_j))
    return fields["p_corr"]


def correct_velocity(
    fields,
    mesh,
    bc_cfg,
    a_p_u,
    a_p_v,
    u_prev,
    v_prev,
    p_prev,
    relax_u,
    relax_v,
    relax_p,
):
    """Apply pressure correction and under-relaxation to velocity and pressure."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    if "p_corr" not in fields:
        raise RuntimeError("Pressure correction field is missing.")

    _apply_pressure_correction_bc(fields["p_corr"], bc_cfg, count_i, count_j)
    grad_p = grad_scalar(fields["p_corr"], mesh)

    volume = mesh["cell_volume"]
    d_u = volume / a_p_u
    d_v = volume / a_p_v

    u_corr = fields["u"][1:-1, 1:-1] - d_u * grad_p[:, :, 0]
    v_corr = fields["v"][1:-1, 1:-1] - d_v * grad_p[:, :, 1]

    fields["u"][1:-1, 1:-1] = u_prev + relax_u * (u_corr - u_prev)
    fields["v"][1:-1, 1:-1] = v_prev + relax_v * (v_corr - v_prev)
    fields["p"][1:-1, 1:-1] = p_prev + relax_p * fields["p_corr"][1:-1, 1:-1]


def update_turbulence(fields, mesh, turb_cfg, bc_cfg, dt, nu, use_rhie_chow=True):
    """Advance turbulence scalars and update nu_t."""
    if not turb_cfg.get("enabled", False) or turb_cfg.get("model") in (None, "none"):
        turbulence.initialize(fields, mesh, turb_cfg)
        return

    interval = int(turb_cfg.get("update_interval", 1))
    if interval > 1:
        counter = turb_cfg.get("_step_counter", 0) + 1
        turb_cfg["_step_counter"] = counter
        if counter % interval != 0:
            return

    model = turb_cfg.get("model")
    if model not in ("k_omega", "sst"):
        raise NotImplementedError(f"Unknown turbulence model: {model}")
    turbulence.initialize(fields, mesh, turb_cfg)
    turbulence.apply_bcs(fields, mesh, bc_cfg, turb_cfg, nu)
    nu_t = turbulence.eddy_viscosity(fields, turb_cfg)
    p_eff = _pressure_effective(fields, turb_cfg)

    nu_eff = nu + nu_t[1:-1, 1:-1]
    a_p = _momentum_diagonal(mesh, nu_eff, dt)
    flux_i, flux_j = _face_mass_flux(
        fields, mesh, a_p, use_rhie_chow=use_rhie_chow, p_field=p_eff
    )

    adv_k = advect_scalar(fields["k"], mesh, flux_i, flux_j)
    adv_omega = advect_scalar(fields["omega"], mesh, flux_i, flux_j)

    grad_u = grad_scalar(fields["u"], mesh)
    grad_v = grad_scalar(fields["v"], mesh)
    source_k, source_other = turbulence.sources(fields, grad_u, grad_v, mesh=mesh)
    nu_k = turbulence.effective_diffusivity(nu, nu_t, field="k", fields=fields)[1:-1, 1:-1]
    nu_other = turbulence.effective_diffusivity(nu, nu_t, field="omega", fields=fields)[1:-1, 1:-1]

    a_k, b_k, _ = _assemble_implicit_system(
        fields["k"], mesh, nu_k, dt, adv_k + source_k, bc_cfg=bc_cfg
    )
    a_other, b_other, _ = _assemble_implicit_system(
        fields["omega"],
        mesh,
        nu_other,
        dt,
        adv_omega + source_other,
        bc_cfg=bc_cfg,
    )

    k_sol = _solve_linear_system(a_k, b_k, use_amg=True)
    other_sol = _solve_linear_system(a_other, b_other, use_amg=True)

    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    k_new = k_sol.reshape((count_i, count_j))
    fields["k"][1:-1, 1:-1] = k_new
    omega_new = other_sol.reshape((count_i, count_j))
    fields["omega"][1:-1, 1:-1] = omega_new
    params = turb_cfg.get("params", {})
    k_min = params.get("k_min", 1.0e-10)
    omega_min = params.get("omega_min", 1.0e-10)
    fields["k"] = np.maximum(fields["k"], k_min)
    fields["omega"] = np.maximum(fields["omega"], omega_min)

    turbulence.apply_bcs(fields, mesh, bc_cfg, turb_cfg, nu)


def post_process(fields, mesh, post_cfg, step, time, post_dir=None):
    """Run configured per-step post-processing hooks."""
    plot_interval = post_cfg.get("plot_interval", 0)
    if not plot_interval:
        return
    if step % plot_interval != 0:
        return
    fields_to_plot = post_cfg.get("fields", ("u", "v", "p"))
    try:
        from post_processing import plot_field_array
    except ImportError:
        return
    for field in fields_to_plot:
        plot_field_array(
            fields,
            mesh,
            field=field,
            step=step,
            time=time,
            output_dir=post_dir,
            show=False,
        )

    line_cfg = post_cfg.get("line_plot", {})
    if line_cfg.get("enabled", False):
        start = line_cfg.get("start")
        end = line_cfg.get("end")
        if start is None or end is None:
            raise ValueError("line_plot requires start and end points.")
        line_fields = line_cfg.get("fields", fields_to_plot)
        try:
            from post_processing import plot_line_array
        except ImportError:
            return
        for field in line_fields:
            plot_line_array(
                fields,
                mesh,
                field=field,
                start=start,
                end=end,
                step=step,
                time=time,
                output_dir=post_dir,
                show=False,
                analytical=line_cfg.get("analytical"),
            )


def continuity_residual(fields, mesh, nu, dt, use_rhie_chow=True, p_field=None):
    """Compute L2 norm of continuity residual."""
    volume = mesh["cell_volume"]
    a_p = _momentum_diagonal(mesh, nu, dt)
    flux_i, flux_j = _face_mass_flux(
        fields, mesh, a_p, use_rhie_chow=use_rhie_chow, p_field=p_field
    )
    div = (
        flux_i[1:, :] - flux_i[:-1, :] + flux_j[:, 1:] - flux_j[:, :-1]
    ) / volume
    return np.sqrt(np.mean(div ** 2))


def log_residual(post_dir, case_name, step, time, inner_iter, residual, mom_u_l2, mom_v_l2):
    """Append residual history to the case log file."""
    post_dir.mkdir(parents=True, exist_ok=True)
    log_path = post_dir / f"{case_name}_residual_log.txt"
    if not log_path.exists():
        log_path.write_text("step time inner_iter continuity_l2 mom_u_l2 mom_v_l2\n")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{step} {time:.6e} {inner_iter} {residual:.6e} {mom_u_l2:.6e} {mom_v_l2:.6e}\n"
        )
    return log_path


def log_timing(post_dir, case_name, step, time, inner_iter, timing):
    """Append timing history to the case log file."""
    post_dir.mkdir(parents=True, exist_ok=True)
    log_path = post_dir / f"{case_name}_timing_log.txt"
    if not log_path.exists():
        log_path.write_text(
            "step time inner_iter total bc adv mom press corr cont turb post save\n"
        )
    line = (
        f"{step} {time:.6e} {inner_iter} "
        f"{timing['total']:.6e} {timing['bc']:.6e} {timing['adv']:.6e} "
        f"{timing['mom']:.6e} {timing['press']:.6e} {timing['corr']:.6e} "
        f"{timing['cont']:.6e} {timing['turb']:.6e} {timing['post']:.6e} "
        f"{timing['save']:.6e}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return log_path


def log_friction(
    post_dir,
    case_name,
    step,
    time,
    cf,
    tau_avg,
    bulk_velocity=None,
    re=None,
    cf_dpdx=None,
):
    """Append skin-friction history to the case log file."""
    post_dir.mkdir(parents=True, exist_ok=True)
    log_path = post_dir / f"{case_name}_friction_log.txt"
    if not log_path.exists():
        log_path.write_text("step time f_darcy f_darcy_dpdx tau_avg u_bulk reynolds\n")
    if bulk_velocity is None:
        bulk_velocity = float("nan")
    if re is None:
        re = float("nan")
    if cf_dpdx is None:
        cf_dpdx = float("nan")
    f_darcy = 4.0 * float(cf)
    f_darcy_dpdx = 4.0 * float(cf_dpdx)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{step} {time:.6e} {f_darcy:.6e} {f_darcy_dpdx:.6e} "
            f"{tau_avg:.6e} {bulk_velocity:.6e} {re:.6e}\n"
        )
    return log_path


def _ensure_single_log_file(post_dir, log_path):
    """Ensure only one residual log exists in the postproc directory."""
    if not post_dir.exists():
        return
    candidates = sorted(post_dir.glob("*_residual_log.txt"))
    others = [p for p in candidates if p.resolve() != log_path.resolve()]
    if others:
        names = ", ".join(p.name for p in others)
        raise ValueError(
            f"Multiple residual logs found in {post_dir}: {names}. Remove extras."
        )


def save_fields(mesh, fields, output_dir, step, time):
    """Save fields to a state .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"fields_{step:06d}.npz"
    payload = {
        "step": np.array(step, dtype=int),
        "time": np.array(time, dtype=float),
        "cell_center": mesh["cell_center"],
    }
    for key, value in fields.items():
        if key == "p_corr":
            continue
        payload[key] = value
    np.savez(save_path, **payload)
    return save_path


def _resolve_restart_state(config, states_dir):
    """Resolve restart file path from config."""
    restart_cfg = config.get("restart", {})
    if not restart_cfg.get("enabled", False):
        return None

    path = restart_cfg.get("path")
    step = restart_cfg.get("step")

    if path:
        return Path(path)
    if step is None:
        raise ValueError("restart enabled but no path or step provided")

    return Path(states_dir) / f"fields_{int(step):06d}.npz"


def _truncate_log(post_dir, case_name, restart_step):
    """Trim residual log entries beyond the restart step."""
    log_path = post_dir / f"{case_name}_residual_log.txt"
    if not log_path.exists():
        return

    lines = log_path.read_text().splitlines()
    if not lines:
        return

    header = lines[0]
    kept = [header]
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        try:
            step = int(float(parts[0]))
        except ValueError:
            continue
        if step < restart_step:
            kept.append(line)
        else:
            break
    log_path.write_text("\n".join(kept) + "\n")


def _resolve_case_paths(config, mesh_path):
    """Resolve case directory paths for mesh, states, and postproc."""
    paths = config.get("paths", {})
    case_dir = Path(paths.get("case_dir", mesh_path.parent.parent))
    mesh_dir = Path(paths.get("mesh_dir", case_dir / "mesh"))
    states_dir = Path(paths.get("states_dir", case_dir / "states"))
    post_dir = Path(paths.get("post_dir", case_dir / "postproc"))
    return case_dir, mesh_dir, states_dir, post_dir


def run_case(config):
    """Run a case configuration through the SIMPLE loop."""
    solver_cfg = config.get("solver", {})
    use_numba = bool(solver_cfg.get("use_numba", False))
    _require_packages(use_numba=use_numba)
    mesh_cfg = config["mesh"]
    mesh_path = Path(mesh_cfg["path"])
    mesh = load_mesh(mesh_cfg)
    time_cfg = config["time"]
    dt = time_cfg.get("dt")
    max_steps = time_cfg["max_steps"]
    cfl = time_cfg.get("cfl")
    cfl_schedule = _parse_value_schedule(time_cfg.get("cfl_schedule"), "cfl_schedule")
    cfl_diff = time_cfg.get("cfl_diff")
    cfl_diff_schedule = _parse_value_schedule(
        time_cfg.get("cfl_diff_schedule"), "cfl_diff_schedule"
    )
    dt_max = time_cfg.get("dt_max", dt)
    dt_max_schedule = _parse_value_schedule(
        time_cfg.get("dt_max_schedule"), "dt_max_schedule"
    )
    dt_min = time_cfg.get("dt_min", 0.0)
    if cfl is None and cfl_schedule is None and dt is None:
        raise ValueError("time.dt must be provided when time.cfl is not set.")
    if (cfl is not None or cfl_schedule is not None) and dt_max is None and dt_max_schedule is None:
        raise ValueError("time.dt_max must be provided when time.cfl is set.")
    nu = config["physics"]["nu"]
    momentum_cfg = solver_cfg.get("momentum", {})
    pressure_cfg = solver_cfg.get("pressure", {})
    use_rhie_chow = solver_cfg.get("rhie_chow", True)
    scheme = solver_cfg.get("scheme", "simple").lower()
    global USE_NUMBA
    if use_numba and not NUMBA_AVAILABLE:
        raise RuntimeError("Numba requested but not available.")
    USE_NUMBA = use_numba
    simple_cfg = solver_cfg.get("simple", {})
    simple_max_iter = simple_cfg.get("max_iter", 10)
    continuity_tol = simple_cfg.get("continuity_tol", 1.0e-4)
    relax_u = simple_cfg.get("relax_u", 0.7)
    relax_v = simple_cfg.get("relax_v", 0.7)
    relax_p = simple_cfg.get("relax_p", 0.3)
    piso_cfg = solver_cfg.get("piso", {})
    piso_correctors = int(piso_cfg.get("n_correctors", 3))
    piso_schedule = _parse_value_schedule(
        piso_cfg.get("corrector_schedule"), "piso.corrector_schedule"
    )
    piso_relax_u = piso_cfg.get("relax_u", relax_u)
    piso_relax_v = piso_cfg.get("relax_v", relax_v)
    piso_relax_p = piso_cfg.get("relax_p", relax_p)
    post_cfg = config.get("post", {})
    save_interval = post_cfg.get("save_interval", 0)
    print_interval = post_cfg.get("print_interval", 1)
    residual_log_interval = int(post_cfg.get("residual_log_interval", 1))
    friction_log_interval = int(post_cfg.get("friction_log_interval", 1))
    timing_cfg = post_cfg.get("timing", {})
    timing_enabled = bool(timing_cfg.get("enabled", False))
    friction_cfg = post_cfg.get("friction", {})
    friction_enabled = bool(friction_cfg.get("enabled", False))

    case_name = config.get("case_name", mesh_path.stem)
    _, _, states_dir, post_dir = _resolve_case_paths(config, mesh_path)
    states_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)
    log_path = post_dir / f"{case_name}_residual_log.txt"
    _ensure_single_log_file(post_dir, log_path)

    restart_path = _resolve_restart_state(config, states_dir)
    if restart_path:
        fields, step, time = load_state(restart_path, mesh, config["initial"])
        if step is None:
            step = int(config.get("restart", {}).get("step", 0) or 0)
        if time is None:
            dt_ref = dt_max if dt_max is not None else (dt or 0.0)
            time = step * dt_ref
        _truncate_log(post_dir, case_name, step)
    else:
        fields = initialize_fields(mesh, config["initial"])
        time = 0.0
        step = 0
        if log_path.exists():
            log_path.unlink()
    turbulence.initialize(fields, mesh, config["turbulence"], config["boundary"])
    step_end = step + max_steps
    start_wall_time = time_module.perf_counter()
    amg_cache = {}
    while step < step_end:
        step_start = time_module.perf_counter()
        timing = {
            "total": 0.0,
            "bc": 0.0,
            "adv": 0.0,
            "mom": 0.0,
            "press": 0.0,
            "corr": 0.0,
            "cont": 0.0,
            "turb": 0.0,
            "post": 0.0,
            "save": 0.0,
        }
        apply_boundary_conditions(fields, mesh, config["boundary"])
        step_cfl = _scheduled_value(step, cfl, cfl_schedule)
        step_cfl_diff = _scheduled_value(step, cfl_diff, cfl_diff_schedule)
        step_dt_max = _scheduled_value(step, dt_max, dt_max_schedule)
        if step_cfl is not None:
            if step_cfl_diff is None:
                step_cfl_diff = step_cfl
            nu_eff_dt = nu
            if "nu_t" in fields:
                nu_eff_dt = nu + fields["nu_t"][1:-1, 1:-1]
            dt_adv = None
            dt_diff = None
            dt = compute_adaptive_dt(
                fields,
                mesh,
                step_cfl,
                step_dt_max,
                dt_min,
                cfl_diff=step_cfl_diff,
                nu_eff=nu_eff_dt,
            )
            if dt <= 0.0:
                break
        cont_res = None
        p_corr_field = fields["p"]
        inner_iter = 0
        mom_u_l2 = float("nan")
        mom_v_l2 = float("nan")
        if scheme == "piso":
            step_correctors = piso_correctors
            if piso_schedule is not None:
                step_correctors = int(
                    _scheduled_value(step, piso_correctors, piso_schedule)
                )
            step_correctors = max(1, step_correctors)
            t0 = time_module.perf_counter()
            apply_boundary_conditions(fields, mesh, config["boundary"])
            timing["bc"] += time_module.perf_counter() - t0

            u_prev = fields["u"][1:-1, 1:-1].copy()
            v_prev = fields["v"][1:-1, 1:-1].copy()
            p_prev = fields["p"][1:-1, 1:-1].copy()

            nu_t = turbulence.eddy_viscosity(fields, config["turbulence"])
            nu_eff = nu + nu_t[1:-1, 1:-1]
            p_eff = fields["p"]

            t0 = time_module.perf_counter()
            adv_rhs = compute_advective_rhs(
                fields, mesh, nu_eff, dt, use_rhie_chow=use_rhie_chow, p_field=p_eff
            )
            timing["adv"] += time_module.perf_counter() - t0

            t0 = time_module.perf_counter()
            mom_diag = solve_momentum(
                fields,
                mesh,
                dt,
                nu_eff,
                adv_rhs,
                config["boundary"],
                solver_cfg=momentum_cfg,
                p_field=p_eff,
                pressure_gradient=config["physics"].get("pressure_gradient", (0.0, 0.0)),
                extra_pressure_field=(
                    (2.0 / 3.0) * fields["k"]
                    if config["turbulence"].get("enabled", False)
                    and "k" in fields
                    else None
                ),
            )
            timing["mom"] += time_module.perf_counter() - t0
            mom_u_l2 = mom_diag.get("mom_u_l2", float("nan"))
            mom_v_l2 = mom_diag.get("mom_v_l2", float("nan"))
            t0 = time_module.perf_counter()
            apply_boundary_conditions(fields, mesh, config["boundary"])
            timing["bc"] += time_module.perf_counter() - t0
            pressure_visc = nu_eff
            a_p_corr = mom_diag["a_p_u"]

            for inner_iter in range(1, step_correctors + 1):
                t0 = time_module.perf_counter()
                solve_pressure(
                    fields,
                    mesh,
                    dt,
                    pressure_visc,
                    config["boundary"],
                    solver_cfg=pressure_cfg,
                    use_rhie_chow=use_rhie_chow,
                    p_field=p_corr_field,
                    a_p_override=a_p_corr,
                    amg_cache=amg_cache,
                )
                timing["press"] += time_module.perf_counter() - t0
                t0 = time_module.perf_counter()
                correct_velocity(
                    fields,
                    mesh,
                    config["boundary"],
                    a_p_corr,
                    a_p_corr,
                    u_prev,
                    v_prev,
                    p_prev,
                    piso_relax_u,
                    piso_relax_v,
                    piso_relax_p,
                )
                timing["corr"] += time_module.perf_counter() - t0
                t0 = time_module.perf_counter()
                apply_boundary_conditions(fields, mesh, config["boundary"])
                timing["bc"] += time_module.perf_counter() - t0

            t0 = time_module.perf_counter()
            cont_res = continuity_residual(
                fields,
                mesh,
                pressure_visc,
                dt,
                use_rhie_chow=use_rhie_chow,
                p_field=p_corr_field,
            )
            timing["cont"] += time_module.perf_counter() - t0
        else:
            for inner_iter in range(1, simple_max_iter + 1):
                t0 = time_module.perf_counter()
                apply_boundary_conditions(fields, mesh, config["boundary"])
                timing["bc"] += time_module.perf_counter() - t0

                u_prev = fields["u"][1:-1, 1:-1].copy()
                v_prev = fields["v"][1:-1, 1:-1].copy()
                p_prev = fields["p"][1:-1, 1:-1].copy()

                nu_t = turbulence.eddy_viscosity(fields, config["turbulence"])
                nu_eff = nu + nu_t[1:-1, 1:-1]
                p_eff = fields["p"]

                t0 = time_module.perf_counter()
                adv_rhs = compute_advective_rhs(
                    fields, mesh, nu_eff, dt, use_rhie_chow=use_rhie_chow, p_field=p_eff
                )
                timing["adv"] += time_module.perf_counter() - t0

                t0 = time_module.perf_counter()
                mom_diag = solve_momentum(
                    fields,
                    mesh,
                    dt,
                    nu_eff,
                    adv_rhs,
                    config["boundary"],
                    solver_cfg=momentum_cfg,
                    p_field=p_eff,
                    pressure_gradient=config["physics"].get("pressure_gradient", (0.0, 0.0)),
                    extra_pressure_field=(
                        (2.0 / 3.0) * fields["k"]
                        if config["turbulence"].get("enabled", False)
                        and "k" in fields
                        else None
                    ),
                )
                timing["mom"] += time_module.perf_counter() - t0
                mom_u_l2 = mom_diag.get("mom_u_l2", float("nan"))
                mom_v_l2 = mom_diag.get("mom_v_l2", float("nan"))
                t0 = time_module.perf_counter()
                apply_boundary_conditions(fields, mesh, config["boundary"])
                timing["bc"] += time_module.perf_counter() - t0
                pressure_visc = nu_eff
                a_p_corr = mom_diag["a_p_u"]
                t0 = time_module.perf_counter()
                solve_pressure(
                    fields,
                    mesh,
                    dt,
                    pressure_visc,
                    config["boundary"],
                    solver_cfg=pressure_cfg,
                    use_rhie_chow=use_rhie_chow,
                    p_field=p_corr_field,
                    a_p_override=a_p_corr,
                    amg_cache=amg_cache,
                )
                timing["press"] += time_module.perf_counter() - t0
                t0 = time_module.perf_counter()
                correct_velocity(
                    fields,
                    mesh,
                    config["boundary"],
                    a_p_corr,
                    a_p_corr,
                    u_prev,
                    v_prev,
                    p_prev,
                    relax_u,
                    relax_v,
                    relax_p,
                )
                timing["corr"] += time_module.perf_counter() - t0

                t0 = time_module.perf_counter()
                apply_boundary_conditions(fields, mesh, config["boundary"])
                timing["bc"] += time_module.perf_counter() - t0

                t0 = time_module.perf_counter()
                cont_res = continuity_residual(
                    fields,
                    mesh,
                    pressure_visc,
                    dt,
                    use_rhie_chow=use_rhie_chow,
                    p_field=p_corr_field,
                )
                timing["cont"] += time_module.perf_counter() - t0
                if cont_res < continuity_tol:
                    break

        t0 = time_module.perf_counter()
        update_turbulence(
            fields, mesh, config["turbulence"], config["boundary"], dt, nu, use_rhie_chow
        )
        timing["turb"] += time_module.perf_counter() - t0

        t0 = time_module.perf_counter()
        post_process(fields, mesh, config["post"], step, time, post_dir=post_dir)
        timing["post"] += time_module.perf_counter() - t0
        if cont_res is not None and (residual_log_interval <= 1 or step % residual_log_interval == 0):
            log_residual(
                post_dir,
                case_name,
                step,
                time,
                inner_iter,
                cont_res,
                mom_u_l2,
                mom_v_l2,
            )
        if friction_enabled and (friction_log_interval <= 1 or step % friction_log_interval == 0):
            try:
                from post_processing import friction_coefficient_array
            except ImportError:
                friction_enabled = False
            else:
                friction = friction_coefficient_array(
                    fields,
                    mesh,
                    boundary=friction_cfg.get("boundary", "j_min"),
                    index_range=friction_cfg.get("range"),
                    nu=config["physics"]["nu"],
                    rho=friction_cfg.get("rho", 1.0),
                    u_ref=friction_cfg.get("u_ref"),
                    use_nu_t=bool(friction_cfg.get("use_nu_t", False)),
                    flow_axis=friction_cfg.get("flow_axis", "i"),
                    hydraulic_diameter=friction_cfg.get("hydraulic_diameter"),
                    pressure_gradient=config["physics"].get("pressure_gradient"),
                )
                log_friction(
                    post_dir,
                    case_name,
                    step,
                    time,
                    friction["cf"],
                    friction["tau_avg"],
                    bulk_velocity=friction.get("bulk_velocity"),
                    re=friction.get("re"),
                    cf_dpdx=friction.get("cf_dpdx"),
                )

        elapsed = time_module.perf_counter() - start_wall_time
        dt_min_report = float(np.min(dt)) if np.ndim(dt) > 0 else float(dt)
        dt_max_report = float(np.max(dt)) if np.ndim(dt) > 0 else float(dt)
        if dt_adv is not None and dt_diff is not None:
            adv_min = float(np.min(dt_adv))
            diff_min = float(np.min(dt_diff))
        else:
            adv_min, diff_min = _dt_limit_estimate(
                fields, mesh, step_cfl, step_cfl_diff, nu_eff_dt
            )
        dt_limit = "adv" if adv_min <= diff_min else "diff"
        if step_dt_max is not None and np.isclose(dt_min_report, float(step_dt_max)):
            dt_limit = "max"
        if dt_min is not None and np.isclose(dt_min_report, float(dt_min)):
            dt_limit = "min"
        resid_value = cont_res if cont_res is not None else float("nan")
        if print_interval is None:
            print_interval = 1
        if print_interval == 0 or step % int(print_interval) == 0:
            print(
                "step {step} time {time:.6e} dt_min {dt_min:.3e} dt_max {dt_max:.3e} "
                "dt_lim {dt_lim} wall {elapsed:.2f}s "
                "scheme {scheme} inner {inner} cont {resid:.6e} mom_u {mom_u:.6e} mom_v {mom_v:.6e}".format(
                    step=step,
                    time=time,
                    dt_min=dt_min_report,
                    dt_max=dt_max_report,
                    dt_lim=dt_limit,
                    elapsed=elapsed,
                    scheme=scheme,
                    inner=inner_iter,
                    resid=resid_value,
                    mom_u=mom_u_l2,
                    mom_v=mom_v_l2,
                ),
                flush=True,
            )
        if save_interval and step % save_interval == 0:
            t0 = time_module.perf_counter()
            save_path = save_fields(mesh, fields, states_dir, step, time)
            timing["save"] += time_module.perf_counter() - t0
            print(f"saved {save_path}", flush=True)

        timing["total"] = time_module.perf_counter() - step_start
        if timing_enabled:
            log_timing(post_dir, case_name, step, time, inner_iter, timing)

        time += dt_min_report
        step += 1

    if timing_enabled and timing_cfg.get("plot", True):
        try:
            from post_processing import plot_timing

            plot_timing(post_dir)
        except Exception:
            pass

    if friction_enabled:
        try:
            from post_processing import plot_friction

            plot_friction(post_dir, output_dir=post_dir)
        except Exception:
            pass

    return fields, step, time
