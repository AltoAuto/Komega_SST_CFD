import numpy as np
import bc

# ---------------------------------------------------------------------------
# Module-level metadata cache
# Stores scalar turbulence state (model name, nu) that must NOT be placed
# inside the fields dict, because bc.apply_periodic_pair iterates over all
# fields.values() and would fail on non-2D (0-d) arrays.
# ---------------------------------------------------------------------------
_turb_state: dict = {
    "model": "k_omega",
    "nu": 1.5e-5,
}

# ---------------------------------------------------------------------------
# Standard Wilcox k-omega constants
# ---------------------------------------------------------------------------

def _k_omega_defaults():
    """Standard Wilcox k-omega coefficients (base model)."""
    return {
        "alpha": 5.0 / 9.0,
        "beta": 3.0 / 40.0,
        "beta_star": 0.09,
        "sigma_k": 0.5,
        "sigma_omega": 0.5,
        # guard lines
        "k_min": 1.0e-15,
        "omega_min": 1.0e-6,
        # Eddy viscosity definition: nu_t = nu_t_coeff * k / omega
        "nu_t_coeff": 1,
        # Near wall condition
        "omega_wall_coeff": 85.0,
    }


# ---------------------------------------------------------------------------
# Menter SST k-omega constants  (Menter 1994; 2003 revision)
# ---------------------------------------------------------------------------

def _sst_defaults():
    """Menter SST k-omega coefficients."""
    return {
        # --- Inner set (k-omega, near-wall) ---
        "sigma_k1": 0.85,
        "sigma_omega1": 0.5,
        "beta1": 0.075,
        "gamma1": 5.0 / 9.0,
        # --- Outer set (k-epsilon transformed, free-stream) ---
        "sigma_k2": 1.0,
        "sigma_omega2": 0.856,
        "beta2": 0.0828,
        "gamma2": 0.44,
        # --- Shared constants ---
        "beta_star": 0.09,
        "a1": 0.31,          # Bradshaw constant for eddy-viscosity limiter
        # --- Guard limits ---
        "k_min": 1.0e-15,
        "omega_min": 1.0e-6,
        # Near-wall omega prescription (same formula as k-omega)
        "omega_wall_coeff": 85.0,
    }


# ---------------------------------------------------------------------------
# SST helper: wall distance
# ---------------------------------------------------------------------------

def _wall_face_centers(mesh, boundary, index_range):
    """Return face-centre coordinates (M, 2) for a boundary segment."""
    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    if boundary in ("i_min", "i_max"):
        max_idx = count_j - 1
        start, end = (0, max_idx) if index_range is None else index_range
        if boundary == "i_min":
            n0 = node[0, start : end + 1, :]
            n1 = node[0, start + 1 : end + 2, :]
        else:
            n0 = node[count_i, start : end + 1, :]
            n1 = node[count_i, start + 1 : end + 2, :]
    else:
        max_idx = count_i - 1
        start, end = (0, max_idx) if index_range is None else index_range
        if boundary == "j_min":
            n0 = node[start : end + 1, 0, :]
            n1 = node[start + 1 : end + 2, 0, :]
        else:
            n0 = node[start : end + 1, count_j, :]
            n1 = node[start + 1 : end + 2, count_j, :]
    return 0.5 * (n0 + n1)


def _compute_wall_distance(mesh, bc_cfg):
    """
    Compute minimum wall distance for every cell (ghosted shape Ni+2, Nj+2).

    Scans bc_cfg for wall-type boundaries and computes the minimum Euclidean
    distance from each cell centre to any wall face centre.  Ghost-layer cells
    are filled by nearest-interior extrapolation.
    """
    cell_center = mesh["cell_center"]   # (Ni, Nj, 2)
    Ni, Nj = cell_center.shape[:2]

    d_int = np.full((Ni, Nj), np.inf)

    for boundary, segments in bc_cfg.items():
        if boundary == "pressure_reference" or segments is None:
            continue
        if isinstance(segments, dict):
            segments = [segments]
        for seg in segments:
            if seg.get("type") != "wall":
                continue
            fc = _wall_face_centers(mesh, boundary, seg.get("range"))  # (M, 2)
            cc_flat = cell_center.reshape(-1, 2)                        # (Ni*Nj, 2)
            diff = cc_flat[:, None, :] - fc[None, :, :]                 # (Ni*Nj, M, 2)
            dist = np.hypot(diff[..., 0], diff[..., 1]).min(axis=1).reshape(Ni, Nj)
            d_int = np.minimum(d_int, dist)

    # If no wall was found (e.g. purely periodic case), use a large value
    d_int[np.isinf(d_int)] = 1.0
    d_int = np.maximum(d_int, 1.0e-12)

    # Expand to ghosted array by nearest-neighbour extrapolation
    d = np.empty((Ni + 2, Nj + 2))
    d[1:-1, 1:-1] = d_int
    d[0,   1:-1] = d[1,   1:-1]
    d[-1,  1:-1] = d[-2,  1:-1]
    d[1:-1,   0] = d[1:-1,   1]
    d[1:-1,  -1] = d[1:-1,  -2]
    d[0, 0]   = d[1, 1];   d[0, -1]  = d[1, -2]
    d[-1, 0]  = d[-2, 1];  d[-1, -1] = d[-2, -2]
    return d


# ---------------------------------------------------------------------------
# SST helper: blending functions
# ---------------------------------------------------------------------------

def _sst_F2(k, omega, d, nu):
    """
    Blending function F2 for the SST eddy-viscosity limiter.

    F2 → 1 in the near-wall (k-omega) region, → 0 in the free stream.
    k, omega, d may be full ghosted arrays or interior-only slices.
    """
    beta_star = 0.09
    sqrt_k = np.sqrt(np.maximum(k, 0.0))
    term1 = 2.0 * sqrt_k / (beta_star * omega * d)
    term2 = 500.0 * nu / (d * d * omega)
    arg2 = np.maximum(term1, term2)
    return np.tanh(arg2 * arg2)


def _sst_F1(k, omega, d, nu, CD_komega):
    """
    Blending function F1 for coefficient blending (near-wall → free-stream).

    F1 = tanh(arg1^4) where arg1 combines wall-distance, viscous-sublayer
    and cross-diffusion length scales.
    """
    beta_star  = 0.09
    sigma_omega2 = 0.856
    sqrt_k = np.sqrt(np.maximum(k, 0.0))
    term1 = sqrt_k / (beta_star * omega * d)
    term2 = 500.0 * nu / (d * d * omega)
    term3 = 4.0 * sigma_omega2 * k / (CD_komega * d * d)
    arg1 = np.minimum(np.maximum(term1, term2), term3)
    return np.tanh(arg1 ** 4)


# ---------------------------------------------------------------------------
# SST helper: finite-difference gradient of a ghosted scalar field
# ---------------------------------------------------------------------------

def _fd_grad(phi_g, mesh):
    """
    Cell-centred gradient of a ghosted scalar using central differences.

    Returns interior-only gradients of shape (Ni, Nj, 2).
    Boundary cells use a one-sided approximation (scaled to match the
    central-difference denominator convention).
    """
    cc = mesh["cell_center"]    # (Ni, Nj, 2)
    Ni, Nj = cc.shape[:2]

    # --- i-direction spacing (denominator for central diff) ---
    dx = np.empty((Ni, Nj))
    if Ni > 2:
        dx[1:-1, :] = cc[2:, :, 0] - cc[:-2, :, 0]
    dx[0,  :] = 2.0 * (cc[1,  :, 0] - cc[0,  :, 0])
    dx[-1, :] = 2.0 * (cc[-1, :, 0] - cc[-2, :, 0])
    dx = np.where(np.abs(dx) < 1.0e-15, 1.0, dx)

    # --- j-direction spacing ---
    dy = np.empty((Ni, Nj))
    if Nj > 2:
        dy[:, 1:-1] = cc[:, 2:, 1] - cc[:, :-2, 1]
    dy[:,  0] = 2.0 * (cc[:,  1, 1] - cc[:,  0, 1])
    dy[:, -1] = 2.0 * (cc[:, -1, 1] - cc[:, -2, 1])
    dy = np.where(np.abs(dy) < 1.0e-15, 1.0, dy)

    dphi_dx = (phi_g[2:, 1:-1] - phi_g[:-2, 1:-1]) / dx
    dphi_dy = (phi_g[1:-1, 2:] - phi_g[1:-1, :-2]) / dy
    return np.stack([dphi_dx, dphi_dy], axis=-1)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize(fields, mesh, cfg, bc_cfg=None):
    """Initialize turbulence fields or zero them when disabled."""
    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        if "nu_t" not in fields:
            count_i = mesh["count_i"].item()
            count_j = mesh["count_j"].item()
            fields["nu_t"] = np.zeros((count_i + 2, count_j + 2), dtype=float)
        else:
            fields["nu_t"][:, :] = 0.0
        return fields

    model = cfg.get("model")
    if model not in ("k_omega", "sst"):
        raise NotImplementedError(f"Unknown turbulence model: {cfg.get('model')}")

    # Validate inlet BCs supply k and omega
    if bc_cfg is not None:
        for boundary, segments in bc_cfg.items():
            if boundary == "pressure_reference" or segments is None:
                continue
            if isinstance(segments, dict):
                segments = [segments]
            for seg in segments:
                if seg.get("type") == "inlet":
                    if ("k" not in seg) or ("omega" not in seg):
                        raise ValueError(
                            "Turbulence inlet requires 'k' and 'omega' values."
                        )

    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    shape = (count_i + 2, count_j + 2)

    if "k" not in fields:
        raise ValueError("Initial k is missing; set it in initial scalars.")
    if "omega" not in fields:
        raise ValueError("Initial omega is missing; set it in initial scalars.")
    if "nu_t" not in fields:
        fields["nu_t"] = np.zeros(shape, dtype=float)

    # Record the active model in the module cache (not in fields, to avoid
    # 0-d array issues in bc.apply_periodic_pair which iterates fields.values())
    _turb_state["model"] = model

    return fields


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _apply_komega_wall(fields, mesh, boundary, index_range, nu, params):
    """Apply wall-resolved k-omega BCs at the first cell centre."""
    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()

    interior, ghost = bc._boundary_indices(boundary, index_range, count_i, count_j)
    _, _, s = bc._boundary_geometry(mesh, boundary, index_range)

    if np.ndim(s) > 1 and s.shape[-1] == 2:
        y = np.linalg.norm(s, axis=-1)
    else:
        y = np.abs(s)

    y_safe = np.maximum(y, 1.0e-12)

    omega_wall = params["omega_wall_coeff"] * nu / (y_safe ** 2)
    omega_wall = np.maximum(omega_wall, params["omega_min"])

    fields["k"][interior]     = params["k_min"]
    fields["omega"][interior] = omega_wall

    bc.set_dirichlet(fields["k"],     mesh, boundary, index_range, params["k_min"])
    bc.set_dirichlet(fields["omega"], mesh, boundary, index_range, omega_wall)


def apply_bcs(fields, mesh, bc_cfg, cfg, nu):
    """Apply turbulence BCs for the active model."""
    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        return

    model = cfg.get("model", "k_omega")
    params = _sst_defaults() if model == "sst" else _k_omega_defaults()

    count_i = mesh["count_i"].item()
    count_j = mesh["count_j"].item()
    range_i, range_j = bc.get_periodic_pairs(bc_cfg, count_i, count_j)
    if range_i is not None:
        bc.apply_periodic_pair(
            {"k": fields["k"], "omega": fields["omega"]}, mesh, "i", range_i
        )
    if range_j is not None:
        bc.apply_periodic_pair(
            {"k": fields["k"], "omega": fields["omega"]}, mesh, "j", range_j
        )

    for boundary, segments in bc_cfg.items():
        if boundary == "pressure_reference" or segments is None:
            continue
        if isinstance(segments, dict):
            segments = [segments]
        for seg in segments:
            bc_type    = seg.get("type")
            index_range = seg.get("range")
            if bc_type == "inlet":
                bc.set_dirichlet(fields["k"],     mesh, boundary, index_range, seg["k"])
                bc.set_dirichlet(fields["omega"], mesh, boundary, index_range, seg["omega"])
            elif bc_type == "wall":
                _apply_komega_wall(fields, mesh, boundary, index_range, nu, params)
            elif bc_type in ("outlet", "symmetry"):
                if "k" in seg:
                    bc.set_dirichlet(fields["k"], mesh, boundary, index_range, seg["k"])
                else:
                    bc.set_neumann(
                        fields["k"], mesh, boundary, index_range, seg.get("k_gradient", 0.0)
                    )
                if "omega" in seg:
                    bc.set_dirichlet(fields["omega"], mesh, boundary, index_range, seg["omega"])
                else:
                    bc.set_neumann(
                        fields["omega"], mesh, boundary, index_range,
                        seg.get("omega_gradient", 0.0),
                    )
            elif bc_type == "periodic":
                continue
            else:
                raise ValueError(f"Unknown BC type for turbulence: {bc_type}")

    # --- SST-only: cache wall distance and nu for use in sources() ---
    if model == "sst":
        if "_d_wall" not in fields or fields["_d_wall"].shape != fields["k"].shape:
            fields["_d_wall"] = _compute_wall_distance(mesh, bc_cfg)
        # Store nu in module cache (not in fields — 0-d arrays break bc.apply_periodic_pair)
        _turb_state["nu"] = float(nu)

        # Compute an initial F2 so eddy_viscosity() can apply the limiter
        # even before sources() has been called for the first time.
        k_s = np.maximum(fields["k"], params["k_min"])
        o_s = np.maximum(fields["omega"], params["omega_min"])
        d   = np.maximum(fields["_d_wall"], 1.0e-12)
        fields["_F2"] = _sst_F2(k_s, o_s, d, float(nu))


# ---------------------------------------------------------------------------
# Eddy viscosity
# ---------------------------------------------------------------------------

def eddy_viscosity(fields, cfg):
    """Compute and store eddy viscosity nu_t from k and omega."""
    if not cfg.get("enabled", False) or cfg.get("model") in (None, "none"):
        fields["nu_t"][:, :] = 0.0
        return fields["nu_t"]

    model = cfg.get("model", "k_omega")

    if model == "sst":
        params    = _sst_defaults()
        k_s       = np.maximum(fields["k"], params["k_min"])
        omega_s   = np.maximum(fields["omega"], params["omega_min"])
        a1        = params["a1"]

        F2 = fields.get("_F2")
        S  = fields.get("_S")

        if F2 is not None and S is not None:
            # Full SST eddy-viscosity limiter: nu_t = a1*k / max(a1*omega, |S|*F2)
            denom = np.maximum(a1 * omega_s, np.asarray(S) * np.asarray(F2))
        else:
            # Fallback before first sources() call (no strain rate cached yet)
            denom = omega_s

        fields["nu_t"][:, :] = np.maximum(a1 * k_s / denom, 0.0)

    else:  # standard k-omega
        params      = _k_omega_defaults()
        k_safe      = np.maximum(fields["k"],     params["k_min"])
        omega_safe  = np.maximum(fields["omega"], params["omega_min"])
        fields["nu_t"][:, :] = params["nu_t_coeff"] * k_safe / omega_safe

    return fields["nu_t"]


# ---------------------------------------------------------------------------
# Source terms
# ---------------------------------------------------------------------------

def sources(fields, grad_u, grad_v, mesh=None):
    """
    Compute turbulence source terms for the active model.

    Parameters
    ----------
    fields : dict
        Contains 'k', 'omega', 'nu_t' (ghosted), and optional SST state.
    grad_u, grad_v : ndarray (Ni, Nj, 2)
        Cell-centred velocity gradients for interior cells.
    mesh : dict or None
        Mesh dict (required for accurate cross-diffusion in SST; if None,
        cross-diffusion is set to zero — adequate for wall-bounded flows).

    Returns
    -------
    source_k, source_omega : ndarray (Ni, Nj)
        Interior-only source term contributions.
    """
    model_tag = _turb_state.get("model", "k_omega")

    if model_tag == "sst":
        return _sources_sst(fields, grad_u, grad_v, mesh)
    return _sources_komega(fields, grad_u, grad_v)


def _sources_komega(fields, grad_u, grad_v):
    """Standard Wilcox k-omega source terms."""
    params     = _k_omega_defaults()
    k          = np.maximum(fields["k"][1:-1, 1:-1],     params["k_min"])
    omega      = np.maximum(fields["omega"][1:-1, 1:-1], params["omega_min"])
    nu_t       = fields["nu_t"][1:-1, 1:-1]

    ux, uy = grad_u[..., 0], grad_u[..., 1]
    vx, vy = grad_v[..., 0], grad_v[..., 1]
    S2 = 2.0 * ux**2 + 2.0 * vy**2 + (uy + vx)**2

    P_k = nu_t * S2

    source_k     = P_k - params["beta_star"] * k * omega
    # Stable form: gamma*S^2 - beta*omega^2  (avoids k in denominator)
    source_omega = params["alpha"] * S2 - params["beta"] * omega**2

    return source_k, source_omega


def _sources_sst(fields, grad_u, grad_v, mesh):
    """
    Menter SST k-omega source terms.

    Computes blending functions F1/F2, the Bradshaw production limiter,
    and the cross-diffusion term.  Caches F1, F2, and |S| in fields for
    use by eddy_viscosity() and effective_diffusivity().
    """
    params  = _sst_defaults()
    k_min   = params["k_min"]
    omega_min = params["omega_min"]

    k     = np.maximum(fields["k"][1:-1, 1:-1],     k_min)
    omega = np.maximum(fields["omega"][1:-1, 1:-1], omega_min)
    nu_t  = fields["nu_t"][1:-1, 1:-1]

    ux, uy = grad_u[..., 0], grad_u[..., 1]
    vx, vy = grad_v[..., 0], grad_v[..., 1]
    S2 = 2.0 * ux**2 + 2.0 * vy**2 + (uy + vx)**2
    S  = np.sqrt(np.maximum(S2, 0.0))

    # Cache ghosted strain-rate magnitude for eddy_viscosity()
    S_g = np.empty_like(fields["k"])
    S_g[1:-1, 1:-1] = S
    S_g[0,   :] = S_g[1,   :];  S_g[-1,  :] = S_g[-2, :]
    S_g[:,   0] = S_g[:,   1];  S_g[:,  -1] = S_g[:,  -2]
    fields["_S"] = S_g

    # Retrieve SST state stored during apply_bcs()
    nu = float(_turb_state.get("nu", 1.5e-5))
    d_wall = fields.get("_d_wall")
    if d_wall is None:
        d_wall = np.ones_like(fields["k"])
    d = np.maximum(d_wall[1:-1, 1:-1], 1.0e-12)

    # --- Cross-diffusion  CD_komega = 2*sigma_omega2 / omega * (∇k · ∇ω) ---
    if mesh is not None:
        grad_k     = _fd_grad(fields["k"],     mesh)   # (Ni, Nj, 2)
        grad_omega = _fd_grad(fields["omega"], mesh)
        CD_cross = (grad_k[..., 0] * grad_omega[..., 0]
                  + grad_k[..., 1] * grad_omega[..., 1])
    else:
        CD_cross = np.zeros_like(k)

    CD_komega = np.maximum(
        2.0 * params["sigma_omega2"] / omega * CD_cross, 1.0e-20
    )

    # --- Blending functions ---
    F1 = _sst_F1(k, omega, d, nu, CD_komega)
    F2 = _sst_F2(k, omega, d, nu)

    # Cache ghosted F1 and F2
    for fname, fi in (("_F1", F1), ("_F2", F2)):
        arr = np.ones_like(fields["k"])
        arr[1:-1, 1:-1] = fi
        arr[0,  :] = arr[1,  :];  arr[-1,  :] = arr[-2, :]
        arr[:,  0] = arr[:,  1];  arr[:, -1] = arr[:, -2]
        fields[fname] = arr

    # --- Blended coefficients ---
    sigma_k     = F1 * params["sigma_k1"]     + (1.0 - F1) * params["sigma_k2"]
    sigma_omega = F1 * params["sigma_omega1"] + (1.0 - F1) * params["sigma_omega2"]
    beta        = F1 * params["beta1"]        + (1.0 - F1) * params["beta2"]
    gamma       = F1 * params["gamma1"]       + (1.0 - F1) * params["gamma2"]

    # Cache blended effective diffusivities (ghosted) for effective_diffusivity()
    nu_t_full = fields["nu_t"]
    for dname, sig in (("_D_k", sigma_k), ("_D_omega", sigma_omega)):
        D_int = nu + sig * nu_t_full[1:-1, 1:-1]
        D_g = np.full_like(fields["k"], nu)
        D_g[1:-1, 1:-1] = D_int
        D_g[0,  :] = D_g[1,  :];  D_g[-1,  :] = D_g[-2, :]
        D_g[:,  0] = D_g[:,  1];  D_g[:, -1] = D_g[:, -2]
        fields[dname] = D_g

    # --- Production with Bradshaw limiter ---
    P_k = nu_t * S2
    P_k = np.minimum(P_k, 20.0 * params["beta_star"] * k * omega)

    # --- k-equation source: Pk - beta* k omega ---
    source_k = P_k - params["beta_star"] * k * omega

    # --- omega-equation source: gamma*S^2 - beta*omega^2 + cross-diffusion ---
    cross_diff   = 2.0 * (1.0 - F1) * params["sigma_omega2"] / omega * CD_cross
    source_omega = gamma * S2 - beta * omega**2 + cross_diff

    return source_k, source_omega


# ---------------------------------------------------------------------------
# Effective diffusivity
# ---------------------------------------------------------------------------

def effective_diffusivity(nu, nu_t, field="k", fields=None):
    """
    Return effective diffusivity for k or omega transport.

    For the SST model, the blended (spatially-varying) sigma is used when
    ``fields`` is supplied and the blended diffusivity has been cached by a
    preceding call to ``sources()``.  Otherwise falls back to the standard
    k-omega constant sigma.

    Parameters
    ----------
    nu : float
        Molecular (kinematic) viscosity.
    nu_t : ndarray
        Eddy viscosity (ghosted shape).
    field : str
        ``"k"`` or ``"omega"``.
    fields : dict or None
        If provided and the SST model is active, returns the precomputed
        blended diffusivity stored in ``_D_k`` / ``_D_omega``.
    """
    # --- SST path: return precomputed blended diffusivity from cache ---
    if fields is not None and _turb_state.get("model") == "sst":
        cache_key = "_D_k" if field == "k" else "_D_omega"
        if cache_key in fields:
            return fields[cache_key]

    # --- Standard k-omega path ---
    params = _k_omega_defaults()
    if field == "k":
        sigma = params["sigma_k"]
    elif field == "omega":
        sigma = params["sigma_omega"]
    else:
        raise ValueError(f"Unsupported field '{field}' for effective_diffusivity.")

    return nu + sigma * nu_t
