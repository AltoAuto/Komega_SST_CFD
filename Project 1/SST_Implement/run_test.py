"""
Short test run: 300 steps with k_omega, then 300 steps with SST.
Each model writes to its own states/<model>/ and postproc/<model>/ subdirectory
so outputs never overwrite each other.
"""
import sys
from pathlib import Path
import numpy as np

# Make project source importable
SRC = Path(__file__).resolve().parents[2] / "Project 1"
sys.path.insert(0, str(SRC))

import solver
from mesh import generate_rect_mesh, make_mesh
import post_processing as pp

# ── Shared paths ────────────────────────────────────────────────────────────
CASE_DIR  = Path(__file__).resolve().parent
MESH_DIR  = CASE_DIR / "mesh"
MESH_PATH = MESH_DIR / "Coarse_mesh.npz"
MESH_DIR.mkdir(parents=True, exist_ok=True)

# ── Physics ─────────────────────────────────────────────────────────────────
LENGTH = 0.02
HEIGHT = 0.05
DPDX   = -0.057
nu     = 1.5e-5
Dh     = 4.0 * HEIGHT
Re     = 1.0e4
U_bulk = Re * nu / Dh
I      = 0.05
L      = 0.07 * Dh
Cmu    = 0.09
k_val     = 1.5 * (U_bulk * I) ** 2
omega_val = np.sqrt(k_val) / (Cmu ** 0.25 * L)
print(f"U_bulk={U_bulk:.4f}  k={k_val:.4e}  omega={omega_val:.4e}")

# ── Mesh ─────────────────────────────────────────────────────────────────────
MESH_CFG = {
    "generator": generate_rect_mesh,
    "params": {
        "length_i": LENGTH,
        "length_j": HEIGHT,
        "count_i": 5,
        "count_j": 20,
        "ratio_i": 1.0,
        "ratio_j": 1.03,
    },
    "path": MESH_PATH,
}
mesh = make_mesh(MESH_CFG)

# Ramped initial u-velocity (wall → symmetry)
cc    = mesh["cell_center"]
y     = cc[:, :, 1]
ramp  = np.clip((y - y.min()) / (y.max() - y.min()), 0.0, 1.0)
u_init = U_bulk * ramp

# ── Boundary conditions (shared) ─────────────────────────────────────────────
BOUNDARY = {
    "i_min": {"type": "periodic", "range": (0, MESH_CFG["params"]["count_j"] - 1)},
    "i_max": {"type": "periodic", "range": (0, MESH_CFG["params"]["count_j"] - 1)},
    "j_min": {"type": "wall",     "range": (0, MESH_CFG["params"]["count_i"] - 1)},
    "j_max": {"type": "symmetry", "range": (0, MESH_CFG["params"]["count_i"] - 1)},
    "pressure_reference": {"cell": (0, 0), "value": 0.0},
}


def run_and_postprocess(model_name):
    """
    Run 300 steps with the given turbulence model and save all outputs to
    model-specific subdirectories so k_omega and sst never share files.
    """
    print(f"\n{'='*60}")
    print(f"  Running model: {model_name.upper()}")
    print(f"{'='*60}")

    # Model-specific output directories
    states_dir = CASE_DIR / "states" / model_name
    post_dir   = CASE_DIR / "postproc" / model_name
    states_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "case_name": f"channel_{model_name}",
        "paths": {
            "case_dir":   CASE_DIR,
            "mesh_dir":   MESH_DIR,
            "states_dir": states_dir,
            "post_dir":   post_dir,
        },
        "mesh":    MESH_CFG,
        "physics": {"nu": nu, "pressure_gradient": (DPDX, 0.0)},
        "time": {
            "cfl": 0.5, "cfl_diff": 0.5,
            "dt_max": 0.1, "dt_min": 0.0,
            "max_steps": 50000,
        },
        "solver": {
            "scheme": "simple",
            "momentum": {"tol": 1e-8, "maxiter": 200},
            "pressure": {"tol": 1e-8, "maxiter": 200},
            "rhie_chow": True,
            "simple": {
                "max_iter": 5,
                "continuity_tol": 1e-4,
                "relax_u": 0.5, "relax_v": 0.5, "relax_p": 0.13,
            },
            "piso": {"n_correctors": 3},
        },
        "restart": {"enabled": False, "step": 0},
        "boundary": BOUNDARY,
        "turbulence": {
            "enabled": True,
            "model": model_name,
            "update_interval": 1,
        },
        "initial": {
            "u": u_init.copy(), "v": 0.0, "p": 0.0,
            "scalars": {"k": k_val, "omega": omega_val},
        },
        "post": {
            "plot_interval":         1000,
            "save_interval":         1000,
            "print_interval":        50,
            "residual_log_interval": 10,
            "friction_log_interval": 10,
            "fields": ("u", "k", "omega"),
            "friction": {
                "enabled": True, "boundary": "j_min", "range": None,
                "u_ref": None, "rho": 1.0, "use_nu_t": False,
                "flow_axis": "i", "hydraulic_diameter": Dh,
            },
            "line_plot": {"enabled": False},
        },
    }

    fields, final_step, final_time = solver.run_case(cfg)
    final_state = solver.save_fields(mesh, fields, states_dir, final_step, final_time)
    print(f"Saved state: {final_state.relative_to(CASE_DIR)}")

    # ── matplotlib plots ─────────────────────────────────────────────────
    pp.plot_field(final_state, MESH_PATH, field="u",     output_dir=post_dir, show=False)
    pp.plot_field(final_state, MESH_PATH, field="k",     output_dir=post_dir, show=False)
    pp.plot_field(final_state, MESH_PATH, field="omega", output_dir=post_dir, show=False)
    pp.plot_yplus(final_state, MESH_PATH, boundary="j_min",
                  nu=nu, output_dir=post_dir)
    pp.plot_convergence(post_dir)
    print("matplotlib plots saved.")

    # ── PyVista plots ────────────────────────────────────────────────────
    try:
        for field in ("speed", "u", "k", "omega"):
            out = pp.pyvista_field(final_state, MESH_PATH, field=field,
                                   output_dir=post_dir, show_edges=True)
            print(f"  PyVista '{field}' -> {out.relative_to(CASE_DIR)}")
        out = pp.pyvista_vectors(final_state, MESH_PATH,
                                 output_dir=post_dir, stride=1)
        print(f"  PyVista vectors -> {out.relative_to(CASE_DIR)}")
    except ImportError as e:
        print(f"PyVista not available: {e}")

    return fields, final_step


# ── Run both models ──────────────────────────────────────────────────────────
fields_ko,  step_ko  = run_and_postprocess("k_omega")
fields_sst, step_sst = run_and_postprocess("sst")

print("\n" + "="*60)
print("  Comparison: final nu_t interior mean")
print(f"  k_omega : {fields_ko['nu_t'][1:-1,1:-1].mean():.4e}")
print(f"  SST     : {fields_sst['nu_t'][1:-1,1:-1].mean():.4e}")
print("\n  Output layout:")
for model in ("k_omega", "sst"):
    print(f"    postproc/{model}/  states/{model}/")
print("="*60)
print("Done.")
