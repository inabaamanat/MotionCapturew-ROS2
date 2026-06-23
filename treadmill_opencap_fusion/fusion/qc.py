"""Quality-control figures so the synchronization and signals can be trusted."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_qc_report(force, cforce, kin, markers, force_gait, sync, merged,
                   out_dir):
    os.makedirs(out_dir, exist_ok=True)
    off = sync.offset_s
    fig, ax = plt.subplots(4, 1, figsize=(11, 14))

    # 1) cross-correlation vs lag
    d = sync.diagnostics
    if "lags_s" in d:
        ax[0].plot(d["lags_s"], d["corr_combined"], color="k", lw=1, label="L+R combined")
        ax[0].plot(d["lags_s"], d["corr_R"], color="g", lw=0.8, alpha=.7, label="R")
        ax[0].plot(d["lags_s"], d["corr_L"], color="orange", lw=0.8, alpha=.7, label="L")
        ax[0].axvline(off, color="r", ls="--", label=f"offset={off:.2f}s")
        ax[0].set_title(f"Sync cross-correlation (peak r={sync.correlation:.2f}, "
                        f"R={sync.corr_right:.2f}, L={sync.corr_left:.2f})")
        ax[0].set_xlabel("force lag (s)"); ax[0].set_ylabel("corr"); ax[0].legend(fontsize=8)

    # 2) shifted overlay: total Fz (force clock -> opencap clock) vs pelvis vert
    t_oc_force = force.t - off
    ax[1].plot(t_oc_force, cforce.total_Fz / merged.meta["bodyweight_N"],
               color="navy", lw=1, label="Total Fz (%BW)")
    ax[1].set_xlim(0, kin.time[-1]); ax[1].set_ylabel("Total Fz (BW)", color="navy")
    ax2 = ax[1].twinx()
    ax2.plot(kin.time, kin.coords["pelvis_ty"], color="crimson", lw=1,
             label="pelvis height")
    ax2.set_ylabel("pelvis_ty (m)", color="crimson")
    ax[1].set_title("Synced overlay: vertical GRF vs pelvis height (OpenCap clock)")
    ax[1].set_xlabel("OpenCap time (s)")

    # 3) per-foot Fz with force heel strikes (shifted to opencap clock)
    for plate, ev, c, lbl in [(cforce.left, force_gait.left, "orange", "L"),
                              (cforce.right, force_gait.right, "g", "R")]:
        ax[2].plot(t_oc_force, plate.Fz / merged.meta["bodyweight_N"], c=c, lw=1,
                   label=f"{lbl} Fz")
        for h in ev.heel_strikes:
            ax[2].axvline(h - off, color=c, ls=":", alpha=.5)
    ax[2].set_xlim(0, kin.time[-1]); ax[2].set_title(
        "Per-foot vertical GRF + force heel strikes (dotted), OpenCap clock")
    ax[2].set_xlabel("OpenCap time (s)"); ax[2].set_ylabel("Fz (BW)"); ax[2].legend(fontsize=8)

    # 4) joint angles
    for col, c in [("knee_angle_l_deg", "orange"), ("knee_angle_r_deg", "g"),
                   ("hip_flexion_l_deg", "tab:orange"), ("hip_flexion_r_deg", "tab:green")]:
        base = col.replace("_deg", "")
        if base in kin.coords:
            ax[3].plot(kin.time, kin.coords[base], c=c, lw=1, label=base)
    ax[3].set_title("Joint angles (OpenCap)"); ax[3].set_xlabel("OpenCap time (s)")
    ax[3].set_ylabel("deg"); ax[3].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    path = os.path.join(out_dir, "qc_report.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
