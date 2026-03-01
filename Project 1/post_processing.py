from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def _resolve_mesh_path(mesh_path_or_dir):
    """Resolve a mesh file path from a file or directory."""
    path = Path(mesh_path_or_dir)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Mesh path not found: {path}")

    candidates = sorted(path.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No .npz mesh files found in: {path}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple .npz meshes found in {path}, pass a file path instead."
        )
    return candidates[0]


def _resolve_state_path(state_path_or_dir):
    """Resolve a state file path from a file or directory."""
    path = Path(state_path_or_dir)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"State path not found: {path}")

    candidates = sorted(path.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No .npz state files found in: {path}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple .npz state files found in {path}, pass a file path instead."
        )
    return candidates[0]


def _resolve_postproc_dir(output_dir=None, state_path=None, mesh_path=None):
    """Determine output directory based on case structure and inputs."""
    if output_dir is not None:
        return Path(output_dir)

    if state_path is not None:
        state_path = Path(state_path)
        if state_path.parent.name == "states":
            return state_path.parent.parent / "postproc"

    if mesh_path is not None:
        mesh_path = Path(mesh_path)
        if mesh_path.parent.name == "mesh":
            return mesh_path.parent.parent / "postproc"

    if state_path is not None:
        return Path(state_path).parent
    if mesh_path is not None:
        return Path(mesh_path).parent
    return Path(".")


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


def _cell_field(state, key, count_i, count_j):
    """Extract a cell-centered field with or without ghost cells."""
    arr = state[key]
    if arr.shape[0] == count_i + 2 and arr.shape[1] == count_j + 2:
        return arr[1:-1, 1:-1]
    if arr.shape[0] == count_i and arr.shape[1] == count_j:
        return arr
    raise ValueError(
        f"Field {key} shape {arr.shape} does not match mesh cells ({count_i}, {count_j})."
    )


def _cell_field_array(arr, count_i, count_j):
    """Extract a cell-centered field array with or without ghost cells."""
    if arr.shape[0] == count_i + 2 and arr.shape[1] == count_j + 2:
        return arr[1:-1, 1:-1]
    if arr.shape[0] == count_i and arr.shape[1] == count_j:
        return arr
    raise ValueError(
        f"Field array shape {arr.shape} does not match mesh cells ({count_i}, {count_j})."
    )


def _boundary_segment(mesh, boundary, index_range, return_length=False):
    """Return boundary geometry and indexing for a segment."""
    node = mesh["node"]
    cell_center = mesh["cell_center"]
    face_center = mesh["face_center"]
    face_normal = mesh["face_normal"]

    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    if boundary in ("i_min", "i_max"):
        start, end = _parse_index_range(index_range, count_j - 1)
        j_idx = np.arange(start, end + 1)
        if boundary == "i_min":
            fc = face_center[0, j_idx, 0, :]
            fn = face_normal[0, j_idx, 0, :]
            cc = cell_center[0, j_idx, :]
            nodes = node[0, start : end + 2, :]
        else:
            n0 = node[count_i, start : end + 1, :]
            n1 = node[count_i, start + 1 : end + 2, :]
            fc = 0.5 * (n0 + n1)
            t = n1 - n0
            fn = np.stack((t[:, 1], -t[:, 0]), axis=-1)
            cc = cell_center[count_i - 1, j_idx, :]
            s = fc - cc
            dot = np.sum(fn * s, axis=-1)
            fn *= np.where(dot < 0.0, -1.0, 1.0)[:, None]
            nodes = node[count_i, start : end + 2, :]

        seg = nodes[1:] - nodes[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        s_coord = np.concatenate(([0.0], np.cumsum(seg_len)))
        s_face = s_coord[:-1] + 0.5 * seg_len
        cell_idx = (np.full_like(j_idx, 0 if boundary == "i_min" else count_i - 1), j_idx)
    elif boundary in ("j_min", "j_max"):
        start, end = _parse_index_range(index_range, count_i - 1)
        i_idx = np.arange(start, end + 1)
        if boundary == "j_min":
            fc = face_center[i_idx, 0, 1, :]
            fn = face_normal[i_idx, 0, 1, :]
            cc = cell_center[i_idx, 0, :]
            nodes = node[start : end + 2, 0, :]
        else:
            n0 = node[start : end + 1, count_j, :]
            n1 = node[start + 1 : end + 2, count_j, :]
            fc = 0.5 * (n0 + n1)
            t = n1 - n0
            fn = np.stack((-t[:, 1], t[:, 0]), axis=-1)
            cc = cell_center[i_idx, count_j - 1, :]
            s = fc - cc
            dot = np.sum(fn * s, axis=-1)
            fn *= np.where(dot < 0.0, -1.0, 1.0)[:, None]
            nodes = node[start : end + 2, count_j, :]

        seg = nodes[1:] - nodes[:-1]
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        s_coord = np.concatenate(([0.0], np.cumsum(seg_len)))
        s_face = s_coord[:-1] + 0.5 * seg_len
        cell_idx = (i_idx, np.full_like(i_idx, 0 if boundary == "j_min" else count_j - 1))
    else:
        raise ValueError("boundary must be one of: i_min, i_max, j_min, j_max")

    fn_mag = np.hypot(fn[:, 0], fn[:, 1])
    n_hat = fn / fn_mag[:, None]
    if return_length:
        return s_face, fc, cc, n_hat, cell_idx, seg_len
    return s_face, fc, cc, n_hat, cell_idx

def plot_mesh(
    mesh_path_or_dir,
    show_cell_centers=False,
    show_face_centers=False,
    line_width=0.7,
    dpi=120,
):
    """Plot the mesh lines and optionally cell/face centers."""
    mesh_path = _resolve_mesh_path(mesh_path_or_dir)
    data = np.load(mesh_path)
    node = data["node"]

    fig, ax = plt.subplots(dpi=dpi)

    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    for i in range(count_i + 1):
        ax.plot(node[i, :, 0], node[i, :, 1], color="k", linewidth=line_width)
    for j in range(count_j + 1):
        ax.plot(node[:, j, 0], node[:, j, 1], color="k", linewidth=line_width)

    if show_cell_centers and "cell_center" in data:
        cc = data["cell_center"]
        ax.scatter(cc[:, :, 0], cc[:, :, 1], s=8, color="tab:blue", marker="o")

    if show_face_centers and "face_center" in data:
        fc = data["face_center"]
        ax.scatter(fc[:, :, 0, 0], fc[:, :, 0, 1], s=8, color="tab:orange", marker="x")
        ax.scatter(fc[:, :, 1, 0], fc[:, :, 1, 1], s=8, color="tab:green", marker="x")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(mesh_path.name)
    plt.tight_layout()

    save_path = mesh_path.parent / f"{mesh_path.stem}_mesh.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()

    return fig, ax, save_path


def plot_field(
    state_path_or_dir,
    mesh_path_or_dir,
    field="speed",
    dpi=120,
    cmap="viridis",
    output_dir=None,
    show=True,
):
    """Plot a scalar field or velocity magnitude on the mesh."""
    state_path = _resolve_state_path(state_path_or_dir)
    mesh_path = _resolve_mesh_path(mesh_path_or_dir)

    mesh = np.load(mesh_path)
    state = np.load(state_path)

    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field(state, "u", count_i, count_j)
        v = _cell_field(state, "v", count_i, count_j)
        phi = np.hypot(u, v)
        label = "velocity magnitude"
    elif key == "u":
        phi = _cell_field(state, "u", count_i, count_j)
        label = "u"
    elif key == "v":
        phi = _cell_field(state, "v", count_i, count_j)
        label = "v"
    elif key == "p":
        phi = _cell_field(state, "p", count_i, count_j)
        label = "p"
    elif key == "k":
        phi = _cell_field(state, "k", count_i, count_j)
        label = "k"
    elif key == "omega":
        phi = _cell_field(state, "omega", count_i, count_j)
        label = "omega"
    else:
        raise ValueError("field must be one of: speed, u, v, p, k, omega")

    fig, ax = plt.subplots(dpi=dpi)
    pcm = ax.pcolormesh(node[:, :, 0], node[:, :, 1], phi, shading="auto", cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(label)
    stats = (
        f"min: {np.nanmin(phi):.3e}\n"
        f"max: {np.nanmax(phi):.3e}\n"
        f"mean: {np.nanmean(phi):.3e}\n"
        f"median: {np.nanmedian(phi):.3e}"
    )
    cbar.ax.text(
        1.6,
        0.5,
        stats,
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        family="monospace",
    )

    title = label
    if "step" in state and "time" in state:
        title = f"{label} | step {int(state['step'])} | time {float(state['time']):.4e}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    post_dir = _resolve_postproc_dir(output_dir, state_path=state_path, mesh_path=mesh_path)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{state_path.stem}_{key}.png"
    fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, save_path


def plot_field_array(
    fields,
    mesh,
    field="speed",
    step=None,
    time=None,
    dpi=120,
    cmap="viridis",
    output_dir=None,
    show=False,
):
    """Plot a scalar field from in-memory arrays."""
    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field_array(fields["u"], count_i, count_j)
        v = _cell_field_array(fields["v"], count_i, count_j)
        phi = np.hypot(u, v)
        label = "velocity magnitude"
    elif key == "u":
        phi = _cell_field_array(fields["u"], count_i, count_j)
        label = "u"
    elif key == "v":
        phi = _cell_field_array(fields["v"], count_i, count_j)
        label = "v"
    elif key == "p":
        phi = _cell_field_array(fields["p"], count_i, count_j)
        label = "p"
    elif key == "k":
        phi = _cell_field_array(fields["k"], count_i, count_j)
        label = "k"
    elif key == "omega":
        phi = _cell_field_array(fields["omega"], count_i, count_j)
        label = "omega"
    else:
        raise ValueError("field must be one of: speed, u, v, p, k, omega")

    fig, ax = plt.subplots(dpi=dpi)
    pcm = ax.pcolormesh(node[:, :, 0], node[:, :, 1], phi, shading="auto", cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(label)
    stats = (
        f"min: {np.nanmin(phi):.3e}\n"
        f"max: {np.nanmax(phi):.3e}\n"
        f"mean: {np.nanmean(phi):.3e}\n"
        f"median: {np.nanmedian(phi):.3e}"
    )
    cbar.ax.text(
        1.6,
        0.5,
        stats,
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        family="monospace",
    )

    title = label
    if step is not None and time is not None:
        title = f"{label} | step {int(step)} | time {float(time):.4e}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=None, mesh_path=None)
    post_dir.mkdir(parents=True, exist_ok=True)
    if step is None:
        save_path = post_dir / f"{key}.png"
    else:
        save_path = post_dir / f"fields_{int(step):06d}_{key}.png"
    fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, save_path


def _plot_poiseuille_profile(ax, line_points, s, analytical):
    """Overlay analytical Poiseuille profile on a line plot."""
    if not analytical:
        return False
    if analytical.get("type", "poiseuille") != "poiseuille":
        return False
    dpdx = analytical.get("dpdx")
    nu = analytical.get("nu")
    if dpdx is None or nu is None:
        return False

    y = line_points[:, 1]
    y0 = float(analytical.get("y0", np.min(y)))
    height = analytical.get("channel_height")
    if height is None:
        height = float(np.max(y) - np.min(y))
    height = float(height)
    if height <= 0.0:
        return False

    y_rel = y - y0
    u_ana = (-float(dpdx) / (2.0 * float(nu))) * y_rel * (height - y_rel)
    ax.plot(s, u_ana, color="tab:orange", linestyle="--", label="analytical")
    return True


def plot_line(
    state_path_or_dir,
    mesh_path_or_dir,
    field,
    start,
    end,
    n_points=200,
    dpi=120,
    cmap="viridis",
    output_dir=None,
    analytical=None,
):
    """Plot a field sampled along a straight line segment."""
    state_path = _resolve_state_path(state_path_or_dir)
    mesh_path = _resolve_mesh_path(mesh_path_or_dir)

    mesh = np.load(mesh_path)
    state = np.load(state_path)

    cell_center = mesh["cell_center"]
    count_i, count_j = cell_center.shape[:2]

    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field(state, "u", count_i, count_j)
        v = _cell_field(state, "v", count_i, count_j)
        phi = np.hypot(u, v)
        label = "velocity magnitude"
    elif key == "u":
        phi = _cell_field(state, "u", count_i, count_j)
        label = "u"
    elif key == "v":
        phi = _cell_field(state, "v", count_i, count_j)
        label = "v"
    elif key == "p":
        phi = _cell_field(state, "p", count_i, count_j)
        label = "p"
    elif key == "k":
        phi = _cell_field(state, "k", count_i, count_j)
        label = "k"
    elif key == "omega":
        phi = _cell_field(state, "omega", count_i, count_j)
        label = "omega"
    else:
        raise ValueError("field must be one of: speed, u, v, p, k, omega")

    points = cell_center.reshape(-1, 2)
    values = phi.reshape(-1)

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    line_points = start[None, :] + (end - start)[None, :] * t[:, None]

    sampled = spi.griddata(points, values, line_points, method="linear")
    if np.any(np.isnan(sampled)):
        missing = np.isnan(sampled)
        sampled_nearest = spi.griddata(points, values, line_points[missing], method="nearest")
        sampled[missing] = sampled_nearest

    s = np.hypot(
        line_points[:, 0] - line_points[0, 0],
        line_points[:, 1] - line_points[0, 1],
    )

    fig, ax = plt.subplots(dpi=dpi)
    show_legend = False
    ax.plot(s, sampled, color="tab:blue", label=label)
    if key == "u":
        show_legend = _plot_poiseuille_profile(ax, line_points, s, analytical)
    ax.set_xlabel("distance along line")
    ax.set_ylabel(label)
    ax.set_title(f"{label} along line")
    ax.grid(True, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend()
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=state_path, mesh_path=mesh_path)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{state_path.stem}_{key}_line.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()
    return fig, ax, save_path


def plot_line_array(
    fields,
    mesh,
    field,
    start,
    end,
    n_points=200,
    dpi=120,
    output_dir=None,
    show=False,
    step=None,
    time=None,
    analytical=None,
):
    """Plot a field along a straight line using in-memory arrays."""
    cell_center = mesh["cell_center"]
    count_i, count_j = cell_center.shape[:2]

    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field_array(fields["u"], count_i, count_j)
        v = _cell_field_array(fields["v"], count_i, count_j)
        phi = np.hypot(u, v)
        label = "velocity magnitude"
    elif key == "u":
        phi = _cell_field_array(fields["u"], count_i, count_j)
        label = "u"
    elif key == "v":
        phi = _cell_field_array(fields["v"], count_i, count_j)
        label = "v"
    elif key == "p":
        phi = _cell_field_array(fields["p"], count_i, count_j)
        label = "p"
    elif key == "k":
        phi = _cell_field_array(fields["k"], count_i, count_j)
        label = "k"
    elif key == "omega":
        phi = _cell_field_array(fields["omega"], count_i, count_j)
        label = "omega"
    else:
        raise ValueError("field must be one of: speed, u, v, p, k, omega")

    points = cell_center.reshape(-1, 2)
    values = phi.reshape(-1)

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    line_points = start[None, :] + (end - start)[None, :] * t[:, None]

    sampled = spi.griddata(points, values, line_points, method="linear")
    if np.any(np.isnan(sampled)):
        missing = np.isnan(sampled)
        sampled_nearest = spi.griddata(points, values, line_points[missing], method="nearest")
        sampled[missing] = sampled_nearest

    s = np.hypot(
        line_points[:, 0] - line_points[0, 0],
        line_points[:, 1] - line_points[0, 1],
    )

    fig, ax = plt.subplots(dpi=dpi)
    show_legend = False
    ax.plot(s, sampled, color="tab:blue", label=label)
    if key == "u":
        show_legend = _plot_poiseuille_profile(ax, line_points, s, analytical)
    ax.set_xlabel("distance along line")
    ax.set_ylabel(label)
    title = f"{label} along line"
    if step is not None and time is not None:
        title = f"{label} | step {int(step)} | time {float(time):.4e}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend()
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=None, mesh_path=None)
    post_dir.mkdir(parents=True, exist_ok=True)
    if step is None:
        save_path = post_dir / f"{key}_line.png"
    else:
        save_path = post_dir / f"fields_{int(step):06d}_{key}_line.png"
    fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, save_path


def friction_coefficient_array(
    fields,
    mesh,
    boundary,
    index_range=None,
    nu=1.0e-3,
    rho=1.0,
    u_ref=None,
    use_nu_t=False,
    flow_axis="i",
    hydraulic_diameter=None,
    pressure_gradient=None,
):
    """Compute skin-friction coefficient from in-memory fields."""
    if u_ref is None:
        if flow_axis == "i":
            u_field = fields["u"][1:-1, 1:-1]
        else:
            u_field = fields["v"][1:-1, 1:-1]
        volume = np.asarray(mesh["cell_volume"])
        if volume.shape != u_field.shape:
            raise ValueError("cell_volume shape does not match velocity field.")
        u_ref = float(np.sum(u_field * volume) / np.sum(volume))
    if u_ref == 0.0 or not np.isfinite(u_ref):
        raise ValueError("u_ref must be nonzero for friction coefficient.")

    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    s_face, fc, cc, n_hat, cell_idx, seg_len = _boundary_segment(
        mesh, boundary, index_range, return_length=True
    )

    u = fields["u"]
    v = fields["v"]
    if u.shape[0] != count_i + 2 or u.shape[1] != count_j + 2:
        raise ValueError("Velocity fields must include ghost cells for friction calc.")

    i_idx, j_idx = cell_idx
    i_int = i_idx + 1
    j_int = j_idx + 1

    if boundary == "i_min":
        i_ghost = np.zeros_like(i_int)
        j_ghost = j_int
    elif boundary == "i_max":
        i_ghost = np.full_like(i_int, count_i + 1)
        j_ghost = j_int
    elif boundary == "j_min":
        i_ghost = i_int
        j_ghost = np.zeros_like(j_int)
    elif boundary == "j_max":
        i_ghost = i_int
        j_ghost = np.full_like(j_int, count_j + 1)
    else:
        raise ValueError("boundary must be one of: i_min, i_max, j_min, j_max")

    t_hat = np.stack((n_hat[:, 1], -n_hat[:, 0]), axis=-1)
    u_cell = u[i_int, j_int]
    v_cell = v[i_int, j_int]
    u_ghost = u[i_ghost, j_ghost]
    v_ghost = v[i_ghost, j_ghost]
    u_t_cell = u_cell * t_hat[:, 0] + v_cell * t_hat[:, 1]
    u_t_ghost = u_ghost * t_hat[:, 0] + v_ghost * t_hat[:, 1]

    dn = np.abs(np.sum((fc - cc) * n_hat, axis=-1))
    dn_safe = np.where(dn > 1.0e-12, dn, 1.0e-12)
    du_t_dn = (u_t_cell - u_t_ghost) / (2.0 * dn_safe)

    nu_eff = float(nu)
    if use_nu_t and "nu_t" in fields:
        nu_eff = nu + fields["nu_t"][i_int, j_int]

    tau = rho * nu_eff * du_t_dn
    length = np.sum(seg_len)
    tau_avg = float(np.sum(tau * seg_len) / length) if length > 0.0 else 0.0
    cf = 2.0 * tau_avg / (float(rho) * float(u_ref) ** 2)
    bulk_velocity = float(u_ref)
    re = None
    if hydraulic_diameter is not None:
        re = float(rho) * bulk_velocity * float(hydraulic_diameter) / float(nu)
    cf_dpdx = None
    if pressure_gradient is not None and hydraulic_diameter is not None:
        if flow_axis == "i":
            dp = float(pressure_gradient[0])
        else:
            dp = float(pressure_gradient[1])
        tau_dp = -dp * float(hydraulic_diameter) / 4.0
        cf_dpdx = 2.0 * tau_dp / (float(rho) * float(u_ref) ** 2)

    return {
        "cf": float(cf),
        "tau_avg": float(tau_avg),
        "bulk_velocity": bulk_velocity,
        "re": re,
        "cf_dpdx": cf_dpdx,
        "tau": tau,
        "s": s_face,
    }


def plot_friction(log_path_or_dir, dpi=120, output_dir=None):
    """Plot skin-friction coefficient history."""
    log_path = Path(log_path_or_dir)
    if log_path.is_dir():
        if (log_path / "postproc").is_dir():
            log_path = log_path / "postproc"
        candidates = sorted(log_path.glob("*_friction_log.txt"))
        if not candidates:
            raise FileNotFoundError(f"No friction log files found in: {log_path}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple friction logs found in {log_path}, pass a file path instead."
            )
        log_path = candidates[0]
    elif not log_path.exists():
        raise FileNotFoundError(f"Friction log file not found: {log_path}")

    data = np.loadtxt(log_path, skiprows=1)
    if data.ndim == 1 and data.size == 4:
        data = data[None, :]

    steps = data[:, 0]
    cf = data[:, 2]
    cf_dpdx = data[:, 3] if data.shape[1] > 3 else None

    fig, ax = plt.subplots(dpi=dpi)
    ax.semilogy(steps, cf, color="tab:purple", marker="o", linestyle="None", label="cf_wall")
    if cf_dpdx is not None:
        ax.semilogy(steps, cf_dpdx, color="tab:orange", marker="o", linestyle="None", label="cf_dpdx")
    ax.set_xlabel("outer step")
    ax.set_ylabel("Darcy friction factor")
    ax.set_title("Darcy friction factor history")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if cf_dpdx is not None:
        ax.legend()
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=log_path, mesh_path=None)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{log_path.stem.replace('_friction_log','')}_friction.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()
    return fig, ax, save_path


def plot_convergence(log_path_or_dir, dpi=120, output_dir=None):
    """Plot continuity and momentum residual histories."""
    log_path = Path(log_path_or_dir)
    if log_path.is_dir():
        if (log_path / "postproc").is_dir():
            log_path = log_path / "postproc"
        candidates = sorted(log_path.glob("*_residual_log.txt"))
        if not candidates:
            raise FileNotFoundError(f"No residual log files found in: {log_path}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple residual logs found in {log_path}, pass a file path instead."
            )
        log_path = candidates[0]
    elif not log_path.exists():
        raise FileNotFoundError(f"Residual log file not found: {log_path}")

    data = np.loadtxt(log_path, skiprows=1)
    if data.ndim == 1 and data.size == 4:
        data = data[None, :]

    steps = data[:, 0]
    residuals = data[:, 3]

    fig, ax = plt.subplots(dpi=dpi)
    ax.semilogy(steps, residuals, color="tab:blue", marker="o", linestyle="None")
    ax.set_xlabel("outer step")
    ax.set_ylabel("continuity L2")
    ax.set_title("Convergence history")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=log_path, mesh_path=None)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{log_path.stem.replace('_residual_log','')}_convergence.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()

    if data.shape[1] >= 6:
        mom_u = data[:, 4]
        mom_v = data[:, 5]

        fig_m, ax_m = plt.subplots(dpi=dpi)
        ax_m.semilogy(steps, mom_u, color="tab:orange", marker="o", linestyle="None", label="u")
        ax_m.semilogy(steps, mom_v, color="tab:green", marker="o", linestyle="None", label="v")
        ax_m.set_xlabel("outer step")
        ax_m.set_ylabel("momentum L2")
        ax_m.set_title("Momentum residual history")
        ax_m.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_m.legend()
        plt.tight_layout()

        save_path_m = (
            post_dir / f"{log_path.stem.replace('_residual_log','')}_momentum.png"
        )
        fig_m.savefig(save_path_m, dpi=dpi)
        plt.show()

    return fig, ax, save_path


def plot_yplus(
    state_path_or_dir,
    mesh_path_or_dir,
    boundary,
    index_range=None,
    nu=1.0e-3,
    rho=1.0,
    dpi=120,
    output_dir=None,
):
    """Plot y+ along a boundary segment using wall-adjacent velocity."""
    state_path = _resolve_state_path(state_path_or_dir)
    mesh_path = _resolve_mesh_path(mesh_path_or_dir)

    mesh = np.load(mesh_path)
    state = np.load(state_path)

    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    s_face, fc, y_plus = _boundary_yplus(
        mesh, state, boundary, index_range, nu, rho, count_i, count_j
    )

    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(s_face, y_plus, marker="o", linestyle="None", color="tab:red")
    ax.set_xlabel("boundary arc length")
    ax.set_ylabel("y+")
    ax.set_title(f"y+ along {boundary}")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=state_path, mesh_path=mesh_path)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{state_path.stem}_yplus_{boundary}.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()
    return fig, ax, save_path


def _boundary_yplus(mesh, state, boundary, index_range, nu, rho, count_i, count_j):
    """Return wall distance and y+ along a boundary segment."""
    s_face, fc, cc, n_hat, cell_idx = _boundary_segment(
        mesh, boundary, index_range
    )

    u = _cell_field(state, "u", count_i, count_j)
    v = _cell_field(state, "v", count_i, count_j)

    u_cell = u[cell_idx]
    v_cell = v[cell_idx]
    t_hat = np.stack((n_hat[:, 1], -n_hat[:, 0]), axis=-1)
    u_t = u_cell * t_hat[:, 0] + v_cell * t_hat[:, 1]

    y = np.abs(np.sum((cc - fc) * n_hat, axis=-1))
    y_safe = np.where(y > 1.0e-12, y, 1.0e-12)
    du_t_dn = u_t / y_safe
    tau_w = rho * nu * du_t_dn
    u_tau = np.sqrt(np.abs(tau_w))
    y_plus = u_tau * y_safe / nu
    return s_face, fc, y_plus


def plot_timing(log_path_or_dir, dpi=120, output_dir=None):
    """Plot per-step timing breakdown from the timing log."""
    log_path = Path(log_path_or_dir)
    if log_path.is_dir():
        if (log_path / "postproc").is_dir():
            log_path = log_path / "postproc"
        candidates = sorted(log_path.glob("*_timing_log.txt"))
        if not candidates:
            raise FileNotFoundError(f"No timing log files found in: {log_path}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple timing logs found in {log_path}, pass a file path instead."
            )
        log_path = candidates[0]
    elif not log_path.exists():
        raise FileNotFoundError(f"Timing log file not found: {log_path}")

    data = np.loadtxt(log_path, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    steps = data[:, 0]
    labels = ["total", "bc", "adv", "mom", "press", "corr", "cont", "turb", "post", "save"]
    series = data[:, 3 : 3 + len(labels)]

    fig, ax = plt.subplots(dpi=dpi)
    for i, label in enumerate(labels):
        ax.plot(steps, series[:, i], marker="o", linestyle="None", label=label)
    ax.set_xlabel("outer step")
    ax.set_ylabel("time [s]")
    ax.set_title("Timing breakdown per step")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    post_dir = _resolve_postproc_dir(output_dir, state_path=log_path, mesh_path=None)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{log_path.stem.replace('_timing_log','')}_timing.png"
    fig.savefig(save_path, dpi=dpi)
    plt.show()
    return fig, ax, save_path


# ===========================================================================
# PyVista visualization  (high-quality 3D-capable rendering)
# ===========================================================================

def _pyvista_import():
    """Lazy import so matplotlib-only users are not forced to install PyVista."""
    try:
        import pyvista as pv
        return pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for pyvista_* functions.  "
            "Install with: pip install pyvista"
        ) from exc


def _pyvista_build_grid(mesh_data, phi, label):
    """Build a PyVista StructuredGrid with a scalar cell-data array."""
    pv = _pyvista_import()
    node = mesh_data["node"]    # (Ni+1, Nj+1, 2)
    x = node[:, :, 0]
    y = node[:, :, 1]
    z = np.zeros_like(x)
    grid = pv.StructuredGrid(x, y, z)
    # VTK StructuredGrid cell ordering: i varies fastest (Fortran / F-order)
    grid.cell_data[label] = phi.flatten(order="F")
    return grid


def _pyvista_resolve_field(state, mesh_data, field):
    """Extract a scalar field array (interior only) and its display label."""
    node = mesh_data["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1
    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field(state, "u", count_i, count_j)
        v = _cell_field(state, "v", count_i, count_j)
        return np.hypot(u, v), "velocity magnitude", key
    field_map = {"u": "u", "v": "v", "p": "p", "k": "k",
                 "omega": "omega", "nu_t": "nu_t"}
    if key not in field_map:
        raise ValueError(
            f"field must be one of: speed, u, v, p, k, omega, nu_t  (got '{field}')"
        )
    return _cell_field(state, key, count_i, count_j), key, key


def pyvista_field(
    state_path_or_dir,
    mesh_path_or_dir,
    field="speed",
    cmap="viridis",
    output_dir=None,
    window_size=(1400, 500),
    show_edges=False,
    show_bounds=True,
):
    """
    Render a 2D scalar field using PyVista and save a high-quality screenshot.

    Produces a *_pv.png file alongside the existing matplotlib outputs.
    All rendering is offscreen so no display is required.

    Parameters
    ----------
    state_path_or_dir : str or Path
        Path to a ``fields_XXXXXX.npz`` state file or its parent ``states/`` dir.
    mesh_path_or_dir : str or Path
        Path to the mesh ``.npz`` file or its parent ``mesh/`` dir.
    field : str
        One of ``speed``, ``u``, ``v``, ``p``, ``k``, ``omega``, ``nu_t``.
    cmap : str
        Matplotlib/PyVista colormap name.
    output_dir : str or Path or None
        Where to save the PNG; defaults to the ``postproc/`` directory.
    window_size : tuple
        Render window size in pixels ``(width, height)``.
    show_edges : bool
        Draw cell edges on the mesh.
    show_bounds : bool
        Add axis grid/bounds annotation.

    Returns
    -------
    Path
        Path to the saved screenshot PNG.
    """
    pv = _pyvista_import()

    state_path = _resolve_state_path(state_path_or_dir)
    mesh_path  = _resolve_mesh_path(mesh_path_or_dir)
    mesh_data  = np.load(mesh_path)
    state      = np.load(state_path)

    phi, label, key = _pyvista_resolve_field(state, mesh_data, field)
    grid = _pyvista_build_grid(mesh_data, phi, label)

    title = label
    if "step" in state and "time" in state:
        title = f"{label}  |  step {int(state['step'])}  |  t = {float(state['time']):.3e}"

    stats = (f"min {np.nanmin(phi):.3e}   max {np.nanmax(phi):.3e}   "
             f"mean {np.nanmean(phi):.3e}")

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        scalars=label,
        cmap=cmap,
        show_edges=show_edges,
        scalar_bar_args={"title": label, "vertical": True,
                         "title_font_size": 12, "label_font_size": 11},
    )
    if show_bounds:
        plotter.show_bounds(grid=True, font_size=9)
    plotter.view_xy()
    plotter.add_title(title, font_size=10)
    plotter.add_text(stats, position="lower_left", font_size=8)

    post_dir = _resolve_postproc_dir(output_dir, state_path=state_path, mesh_path=mesh_path)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{state_path.stem}_{key}_pv.png"
    plotter.screenshot(str(save_path))
    plotter.close()
    return save_path


def pyvista_field_array(
    fields,
    mesh,
    field="speed",
    step=None,
    time=None,
    output_dir=None,
    cmap="viridis",
    window_size=(1400, 500),
    show_edges=False,
    show_bounds=True,
):
    """
    Render a 2D scalar field from in-memory arrays using PyVista.

    Same as :func:`pyvista_field` but operates on live solver arrays rather
    than saved ``.npz`` files.  Useful for in-situ post-processing.

    Parameters
    ----------
    fields : dict
        Solver field dict (ghosted arrays).
    mesh : dict
        Loaded mesh dict.
    field : str
        One of ``speed``, ``u``, ``v``, ``p``, ``k``, ``omega``, ``nu_t``.
    step, time : int/float or None
        Appended to the title if provided.
    output_dir : str or Path or None
        Output directory; defaults to ``"."`` when None.
    """
    pv = _pyvista_import()

    node = mesh["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    key = field.lower()
    if key in ("speed", "vmag", "velocity"):
        u = _cell_field_array(fields["u"], count_i, count_j)
        v = _cell_field_array(fields["v"], count_i, count_j)
        phi = np.hypot(u, v)
        label = "velocity magnitude"
    else:
        field_map = {"u": "u", "v": "v", "p": "p", "k": "k",
                     "omega": "omega", "nu_t": "nu_t"}
        if key not in field_map:
            raise ValueError(
                f"field must be one of: speed, u, v, p, k, omega, nu_t  (got '{field}')"
            )
        phi  = _cell_field_array(fields[key], count_i, count_j)
        label = key

    grid = _pyvista_build_grid(mesh, phi, label)

    title = label
    if step is not None and time is not None:
        title = f"{label}  |  step {int(step)}  |  t = {float(time):.3e}"

    stats = (f"min {np.nanmin(phi):.3e}   max {np.nanmax(phi):.3e}   "
             f"mean {np.nanmean(phi):.3e}")

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        scalars=label,
        cmap=cmap,
        show_edges=show_edges,
        scalar_bar_args={"title": label, "vertical": True,
                         "title_font_size": 12, "label_font_size": 11},
    )
    if show_bounds:
        plotter.show_bounds(grid=True, font_size=9)
    plotter.view_xy()
    plotter.add_title(title, font_size=10)
    plotter.add_text(stats, position="lower_left", font_size=8)

    post_dir = Path(output_dir) if output_dir is not None else Path(".")
    post_dir.mkdir(parents=True, exist_ok=True)
    if step is None:
        save_path = post_dir / f"{key}_pv.png"
    else:
        save_path = post_dir / f"fields_{int(step):06d}_{key}_pv.png"

    plotter.screenshot(str(save_path))
    plotter.close()
    return save_path


def pyvista_vectors(
    state_path_or_dir,
    mesh_path_or_dir,
    output_dir=None,
    cmap="coolwarm",
    scale=None,
    stride=1,
    window_size=(1400, 500),
):
    """
    Render velocity vector arrows using PyVista glyphs.

    Parameters
    ----------
    state_path_or_dir : str or Path
        Path to a ``fields_XXXXXX.npz`` state file or its ``states/`` dir.
    mesh_path_or_dir : str or Path
        Path to the mesh ``.npz`` file or its ``mesh/`` dir.
    output_dir : str or Path or None
        Output directory; defaults to the ``postproc/`` directory.
    cmap : str
        Colormap for arrow colouring (by velocity magnitude).
    scale : float or None
        Arrow scale factor.  If ``None``, PyVista auto-scales by magnitude.
    stride : int
        Sub-sample every *stride* cells in each direction to reduce clutter
        on dense meshes.  ``stride=1`` plots every cell.
    window_size : tuple
        Render window pixel dimensions ``(width, height)``.

    Returns
    -------
    Path
        Path to the saved screenshot PNG.
    """
    pv = _pyvista_import()

    state_path = _resolve_state_path(state_path_or_dir)
    mesh_path  = _resolve_mesh_path(mesh_path_or_dir)
    mesh_data  = np.load(mesh_path)
    state      = np.load(state_path)

    node    = mesh_data["node"]
    count_i = node.shape[0] - 1
    count_j = node.shape[1] - 1

    u_int = _cell_field(state, "u", count_i, count_j)   # (Ni, Nj)
    v_int = _cell_field(state, "v", count_i, count_j)
    speed = np.hypot(u_int, v_int)
    cc    = mesh_data["cell_center"]                     # (Ni, Nj, 2)

    # Sub-sample with stride
    sl = (slice(None, None, stride), slice(None, None, stride))
    cc_s    = cc[sl]
    u_s     = u_int[sl]
    v_s     = v_int[sl]
    speed_s = speed[sl]

    n_pts = cc_s.shape[0] * cc_s.shape[1]
    points  = np.zeros((n_pts, 3))
    points[:, 0] = cc_s[:, :, 0].flatten(order="F")
    points[:, 1] = cc_s[:, :, 1].flatten(order="F")

    vectors = np.zeros((n_pts, 3))
    vectors[:, 0] = u_s.flatten(order="F")
    vectors[:, 1] = v_s.flatten(order="F")

    cloud = pv.PolyData(points)
    cloud["vectors"] = vectors
    cloud["speed"]   = speed_s.flatten(order="F")

    glyph_kw = dict(orient="vectors", scale="speed", factor=1.0)
    if scale is not None:
        glyph_kw = dict(orient="vectors", scale=False, factor=float(scale))
    glyphs = cloud.glyph(**glyph_kw)

    title = "velocity vectors"
    if "step" in state and "time" in state:
        title = f"velocity vectors  |  step {int(state['step'])}  |  t = {float(state['time']):.3e}"

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.add_mesh(
        glyphs,
        scalars="speed",
        cmap=cmap,
        scalar_bar_args={"title": "|U|", "vertical": True},
    )
    plotter.view_xy()
    plotter.add_title(title, font_size=10)

    post_dir = _resolve_postproc_dir(output_dir, state_path=state_path, mesh_path=mesh_path)
    post_dir.mkdir(parents=True, exist_ok=True)
    save_path = post_dir / f"{state_path.stem}_vectors_pv.png"
    plotter.screenshot(str(save_path))
    plotter.close()
    return save_path
