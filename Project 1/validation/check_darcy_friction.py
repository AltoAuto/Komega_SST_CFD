import numpy as np
from pathlib import Path

LOG_KO = Path(r"C:\Users\altoa\Desktop\CHT\Project\Project 1\CFD\Project 1\SST_Implement\postproc\k_omega\channel_k_omega_friction_log.txt")
LOG_SST = Path(r"C:\Users\altoa\Desktop\CHT\Project\Project 1\CFD\Project 1\SST_Implement\postproc\sst\channel_sst_friction_log.txt")

def f_darcy_haaland(Re, eD=0.0):
    """Darcy friction factor using Haaland; smooth if eD=0."""
    Re = float(abs(Re))
    if Re <= 0:
        return np.nan
    if Re < 2300:
        return 64.0 / Re
    inv_sqrt_f = -1.8 * np.log10((eD / 3.7) ** 1.11 + 6.9 / Re)
    return 1.0 / (inv_sqrt_f ** 2)

def load_log(path: Path):
    d = np.genfromtxt(path, names=True)
    # Ensure required columns exist
    required = ["step", "time", "f_darcy", "f_darcy_dpdx", "tau_avg", "u_bulk", "reynolds"]
    missing = [c for c in required if c not in d.dtype.names]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return d

def tail_stats(d, N=200, eD=0.0):
    N = min(N, len(d))
    sl = slice(-N, None)

    Re_end = np.mean(np.abs(d["reynolds"][sl]))
    f_wall_end = np.mean(np.abs(d["f_darcy"][sl]))
    f_dpdx_end = np.mean(np.abs(d["f_darcy_dpdx"][sl]))
    tau_end = np.mean(np.abs(d["tau_avg"][sl]))
    u_bulk_end = np.mean(np.abs(d["u_bulk"][sl]))

    f_ref = f_darcy_haaland(Re_end, eD=eD)

    rel_diff = np.nan if f_wall_end == 0 else abs(f_wall_end - f_dpdx_end) / abs(f_wall_end)
    rel_err_ref = np.nan if np.isnan(f_ref) or f_ref == 0 else (f_wall_end - f_ref) / f_ref

    last = d[-1]
    last_row = {
        "step": int(last["step"]),
        "time": float(last["time"]),
        "Re": float(last["reynolds"]),
        "u_bulk": float(last["u_bulk"]),
        "f_wall": float(last["f_darcy"]),
        "f_dpdx": float(last["f_darcy_dpdx"]),
        "tau": float(last["tau_avg"]),
    }

    summary = {
        "N_tail": N,
        "Re_end": Re_end,
        "u_bulk_end": u_bulk_end,
        "tau_end": tau_end,
        "f_wall_end": f_wall_end,
        "f_dpdx_end": f_dpdx_end,
        "rel_diff_wall_vs_dpdx": rel_diff,
        "f_ref_haaland": f_ref,
        "rel_err_vs_ref": rel_err_ref,
    }
    return last_row, summary

def print_report(name, last_row, s):
    print(f"\n=== {name} ===")
    print(f"Last step={last_row['step']}  time={last_row['time']:.6e}  Re={last_row['Re']:.6g}  u_bulk={last_row['u_bulk']:.6g}")
    print(f"  f_wall={last_row['f_wall']:.6g}  f_dpdx={last_row['f_dpdx']:.6g}  tau_avg={last_row['tau']:.6g}")
    print(f"Tail avg over last {s['N_tail']} points:")
    print(f"  Re_end     = {s['Re_end']:.6g}")
    print(f"  u_bulk_end = {s['u_bulk_end']:.6g}")
    print(f"  f_wall_end = {s['f_wall_end']:.6g}")
    print(f"  f_dpdx_end = {s['f_dpdx_end']:.6g}")
    print(f"  |f_wall - f_dpdx|/|f_wall| = {s['rel_diff_wall_vs_dpdx']*100:.3f}%")
    print(f"  f_ref (Haaland, e/D=0)     = {s['f_ref_haaland']:.6g}")
    print(f"  (f_wall - f_ref)/f_ref     = {s['rel_err_vs_ref']*100:.3f}%")

# ---- main ----
N = 200
eD = 0.0

d_ko = load_log(LOG_KO)
d_sst = load_log(LOG_SST)

last_ko, sum_ko = tail_stats(d_ko, N=N, eD=eD)
last_sst, sum_sst = tail_stats(d_sst, N=N, eD=eD)

print_report("k-omega", last_ko, sum_ko)
print_report("SST", last_sst, sum_sst)

# Side-by-side comparison (key metrics)
print("\n=== Comparison (tail averages) ===")
print(f"Re:     k-omega={sum_ko['Re_end']:.6g}   SST={sum_sst['Re_end']:.6g}")
print(f"f_wall: k-omega={sum_ko['f_wall_end']:.6g}   SST={sum_sst['f_wall_end']:.6g}")
print(f"err% vs Haaland: k-omega={sum_ko['rel_err_vs_ref']*100:.3f}%   SST={sum_sst['rel_err_vs_ref']*100:.3f}%")
print(f"|wall-dpdx|/wall: k-omega={sum_ko['rel_diff_wall_vs_dpdx']*100:.3f}%   SST={sum_sst['rel_diff_wall_vs_dpdx']*100:.3f}%")