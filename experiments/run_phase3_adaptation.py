"""
Phase III Experiment: Error-Adaptive Dynamic Memory Representation K(t).

Demonstrates the error-controlled mechanism on the nonlinear Fractional Duffing Oscillator.
Monitors:
- Trajectory evolution (x_1(t), x_2(t))
- Active memory modes K(t) across dynamical regimes
- Estimated memory error \\widehat{\\epsilon}_M(t) vs prescribed tolerance \\epsilon_{tol}
- Clean, non-squashed phase portrait on independent spatial axes
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.systems import FractionalDuffing
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


def run_phase3_experiment(
    beta: float = 0.85,
    T: float = 6.0,
    num_steps: int = 300,
    tol: float = 1.0e-3,
    save_dir: str = "results",
):
    os.makedirs(save_dir, exist_ok=True)
    from experiments.archive_utils import archive_previous_results
    archive_previous_results("phase3_adaptation", ["phase3_dynamic_memory_adaptation.png", "phase3_dynamic_memory_adaptation_data.json"], save_dir=save_dir)

    t_grid = torch.linspace(0.0, T, num_steps + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

    duffing = FractionalDuffing(beta=beta)

    print("\n" + "=" * 78)
    print(f"  PHASE III: DYNAMIC MEMORY ADAPTATION ON FRACTIONAL DUFFING (beta={beta})")
    print(f"  Simulation Horizon: T = {T:.1f}, Steps N = {num_steps}, tol = {tol:.1e}")
    print("=" * 78)

    # 1. Run AdaMem-FDE Adaptive Solver
    solver_adapt = AdaptiveSOEFDESolver(
        beta=beta,
        tol=tol,
        K_init=8,
        K_min=8,
        K_max=24,
        delta_K=4,
        prune_ratio=0.35,
        patience=6,
    )
    t0 = time.perf_counter()
    sol = solver_adapt.solve(duffing.rhs, z0, t_grid)
    runtime = (time.perf_counter() - t0) * 1000.0

    print(f"Average Active Modes:      \\bar{{K}} = {sol.avg_modes:.2f}")
    print(f"Maximum Active Modes:      K_max = {sol.max_modes}")
    print(f"Total Adaptation Events:   N_adapt = {sol.num_adaptations}")
    print(f"Runtime:                   {runtime:.2f} ms")
    print("-" * 78)

    # Breakdown of events
    expansions = [ev for ev in sol.events if ev.event_type == "expand"]
    prunes = [ev for ev in sol.events if ev.event_type == "prune"]
    print(f"Mode Expansions: {len(expansions)}, Mode Prunings: {len(prunes)}")
    for i, ev in enumerate(sol.events):
        print(
            f"  Event {i+1:02d} at t={ev.t:5.2f} (step {ev.step_idx:3d}): "
            f"{ev.K_old} -> {ev.K_new} modes ({ev.event_type.upper()}), err={ev.error_est:.3e}"
        )
    print("=" * 78)

    # Visualization: 2-column layout eliminating the squashed phase portrait bug
    t_np = t_grid.cpu().numpy()
    z_np = sol.z.cpu().numpy()
    modes_np = np.array(sol.modes_history)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])

    # Left Top: State Trajectories vs Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_np, z_np[:, 0], "b-", lw=1.8, label=r"$x_1(t)$ (Displacement)")
    ax1.plot(t_np, z_np[:, 1], "r--", lw=1.5, label=r"$x_2(t)$ (Velocity)")
    ax1.set_ylabel("State Variables", fontsize=10)
    ax1.set_title("(a) Fractional Duffing State Trajectories", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Left Bottom: Memory Modes K(t) vs Time (sharing time axis with ax1)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.step(t_np, modes_np, "g-", lw=2.0, where="post", label=rf"Active Modes $K(t)$ ($\bar{{K}}={sol.avg_modes:.1f}$)")
    ax2.axhline(sol.avg_modes, color="teal", linestyle="--", lw=1.5, alpha=0.8, label=rf"Mean $\bar{{K}} = {sol.avg_modes:.1f}$")
    ax2.axhline(24, color="gray", linestyle=":", lw=1.2, label=r"Cap Ceiling $K_{\max}=24$")
    for ev in expansions:
        ax2.axvline(ev.t, color="magenta", linestyle=":", alpha=0.7, label="Expansion Event" if ev == expansions[0] else "")
    ax2.set_xlabel("Time $t$", fontsize=10)
    ax2.set_ylabel(r"Modes $K(t)$", fontsize=10)
    ax2.set_title("(b) Dynamic Memory Mode Allocation", fontsize=11, fontweight="bold")
    ax2.set_ylim(4, 28)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    # Right: Phase Portrait on independent spatial axes (NOT sharing time axis!)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(z_np[:, 0], z_np[:, 1], color="purple", lw=1.8, label="Duffing Orbit")
    ax3.plot(z0[0].item(), z0[1].item(), "go", ms=8, label="Initial State $(x_0, \dot{x}_0)$")
    ax3.set_xlabel(r"Displacement $x_1$", fontsize=11)
    ax3.set_ylabel(r"Velocity $x_2$", fontsize=11)
    ax3.set_title("(c) Phase Portrait $(x_1 \mathrm{\ vs\ } x_2)$", fontsize=11, fontweight="bold")
    ax3.set_xlim(-1.6, 1.6)
    ax3.set_ylim(-1.6, 1.6)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "phase3_dynamic_memory_adaptation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Save structured benchmark data for research paper reporting
    import json
    data_record = {
        "beta": beta,
        "T": T,
        "num_steps": num_steps,
        "tol": tol,
        "runtime_ms": float(runtime),
        "avg_modes": float(sol.avg_modes),
        "max_modes": int(sol.max_modes),
        "num_adaptations": int(sol.num_adaptations),
        "expansion_events": [{"step": ev.step_idx, "t": float(ev.t), "K_old": ev.K_old, "K_new": ev.K_new} for ev in expansions],
    }
    json_path = os.path.join(save_dir, "phase3_dynamic_memory_adaptation_data.json")
    with open(json_path, "w") as f:
        json.dump(data_record, f, indent=2)

    print(f"\n[Artifact Saved] Phase III adaptation figure saved to {plot_path}")
    print(f"[Artifact Saved] Phase III adaptation data saved to {json_path}\n")


if __name__ == "__main__":
    run_phase3_experiment()
