from pathlib import Path
import sys
import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import solver
from mesh import generate_rect_mesh, make_mesh
from post_processing import plot_mesh, plot_field, plot_convergence, plot_yplus, plot_line
from visualization.vis_mesh import view_mesh_pyvista

CASE_NAME = "komega_channel"

CASE_DIR = Path(__file__).resolve().parent
MESH_DIR = CASE_DIR / "mesh"
STATES_DIR = CASE_DIR / "states"
POST_DIR = CASE_DIR / "postproc"
MESH_PATH = MESH_DIR / f"{CASE_NAME}.npz"

LENGTH = 0.02
HEIGHT = 0.05

#-------------------------
# Re = 8158 -> DPDX = 0.04
#-------------------------

DPDX = -0.057
DPDY = 0
nu = 1.5e-5

PHYSICS = {
    "nu": nu, # kinematic viscosity
    "pressure_gradient": (DPDX, DPDY),
}

MESH = {
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

# max step -> 100000
# cfl_diff -> 0.2
TIME = {
    "cfl": 0.5,
    "cfl_diff":0.5,
    "dt_max":0.1,
    "dt_min": 0.0,
    "max_steps":50000,
}

RESTART = {
    "enabled": False,
    "step": 1145,
}

SOLVER = {
    "scheme": "simple",
    "momentum": {
        "tol": 1.0e-8,
        "maxiter": 200,
    },
    "pressure": {
        "tol": 1.0e-8,
        "maxiter": 200,
    },
    "rhie_chow": True,
    "simple": {
        "max_iter": 5,
        "continuity_tol": 1.0e-4,
        "relax_u": 0.5,
        "relax_v": 0.5,
        "relax_p": 0.13,
    },
    "piso": {
        "n_correctors": 3,
    }
}

POST = {
    "plot_interval": 5000,
    "save_interval": 5000,
    "print_interval": 100,
    "residual_log_interval":10,
    "friction_log_interval":10,
    "fields": ("u", "k","omega"),
    "line_plot": {
        "enabled": True,
        "start": (0.01, 0.0),
        "end": (0.01, 0.05),
        "fields": ("u",),
        "analytical": {
            "type": "poiseuille",
            "field": "u",
            "dpdx": DPDX,
            "nu": PHYSICS["nu"],
            "channel_height": 2.0 * HEIGHT,
            "y0": 0.0,
        },
    },
    "friction": {
        "enabled": True,
        "boundary": "j_min",
        "range": None,
        "u_ref": None,
        "rho": 1.0,
        "use_nu_t": False,
        "flow_axis": "i",
        "hydraulic_diameter": 4.0 * HEIGHT,
    },
}

BOUNDARY = {
    "i_min": {
        "type": "periodic",
        "range": (0, MESH["params"]["count_j"] - 1),
    },
    "i_max": {
        "type": "periodic",
        "range": (0, MESH["params"]["count_j"] - 1),
    },
    "j_min": {
        "type": "wall",
        "range": (0, MESH["params"]["count_i"] - 1),
    },
    "j_max": {
        "type": "symmetry",
        "range": (0, MESH["params"]["count_i"] - 1),
    },
    "pressure_reference": {"cell": (0, 0), "value": 0.0},
}

# --- Turbulence Boundary Calculation ---
nu = 1.5e-5
Dh = 4.0 * HEIGHT
Re_target = 1.0e4
U_bulk = Re_target * nu / Dh

I = 0.05    # turbulence intensity: Re -> 10^4
L = 0.07 * Dh
Cmu = 0.09
k_val = 1.5 * (U_bulk * I)**2
omega_val = np.sqrt(k_val) / (Cmu**0.25 * L)
print(U_bulk, k_val, omega_val)
#--------------------------------------

INITIAL = {
    "u": U_bulk,
    "v": 0.0,
    "p": 0.0,
    "scalars": {
        "k": k_val,
        "omega": omega_val
    },
}

TURBULENCE = {
    "enabled": True,
    "model": "k_omega",
    "update_interval": 1,
}

CONFIG = {
    "case_name": CASE_NAME,
    "paths": {
        "case_dir": CASE_DIR,
        "mesh_dir": MESH_DIR,
        "states_dir": STATES_DIR,
        "post_dir": POST_DIR,
    },
    "mesh": MESH,
    "time": TIME,
    "physics": PHYSICS,
    "solver": SOLVER,
    "restart": RESTART,
    "turbulence": TURBULENCE,
    "post": POST,
    "boundary": BOUNDARY,
    "initial": INITIAL,
}

if __name__ == "__main__":
    for path in (MESH_DIR, STATES_DIR, POST_DIR):
        path.mkdir(parents=True, exist_ok=True)

    mesh = make_mesh(MESH)

    view_mesh_pyvista(MESH["path"])    #view mesh through pyvista

    plot_mesh(MESH_PATH, line_width=0.6, dpi=140)

    # ramp initial condition
    cell_center = mesh["cell_center"]
    y = cell_center[:, :, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    u_top = float(INITIAL["u"])
    ramp_height = y_max - y_min
    ramp = (y - y_min) / ramp_height
    ramp = np.clip(ramp, 0.0, 1.0)
    u_init = u_top * ramp
    CONFIG["initial"]["u"] = u_init

    fields, final_step, final_time = solver.run_case(CONFIG)

    final_state = solver.save_fields(mesh, fields, STATES_DIR, final_step, final_time)

    plot_convergence(POST_DIR)
    plot_field(final_state, MESH_PATH, field="u", output_dir=POST_DIR)
    plot_field(final_state, MESH_PATH, field="v", output_dir=POST_DIR)
    plot_field(final_state, MESH_PATH, field="p", output_dir=POST_DIR)
    plot_yplus(final_state, MESH_PATH, boundary="j_min", nu=PHYSICS["nu"], output_dir=POST_DIR)

    if POST.get("line_plot", {}).get("enabled", False):
        line_cfg = POST["line_plot"]
        for field in line_cfg.get("fields", POST["fields"]):
            plot_line(
                final_state,
                MESH_PATH,
                field=field,
                start=line_cfg["start"],
                end=line_cfg["end"],
                output_dir=POST_DIR,
                analytical=line_cfg.get("analytical"),
            )
