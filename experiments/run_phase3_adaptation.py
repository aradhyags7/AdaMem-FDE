"""
Phase III Experiment: Error-Adaptive Dynamic Memory Representation K(t).

Demonstrates the error-controlled mechanism on the nonlinear Fractional Duffing Oscillator.
Monitors:
- Trajectory evolution (x_1(t), x_2(t))
- Active memory modes K(t) across dynamical regimes
- Estimated memory error \widehat{\epsilon}_M(t) vs prescribed tolerance \epsilon_{tol}
- Adaptation events (mode expansion and pruning)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.systems import FractionalDuffing
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver
from core.solvers.fixed_soe import FixedSOEFDESolver


def run_phase3_experiment(
    beta: float = 0.85,
    T: float = 12.0,
    num_steps: int = 600,
    tol: float = 5e-4,
    save_dir: str = "results",
):
    os.makedirs(save_dir, exist_ok=True)
    t_grid = torch.linspace(0.0, T, num_steps + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

    duffing = FractionalDuffing(beta=beta)

    print("\n" + "=" * 78)
    print(f"  PHASE III: DYNAMIC MEMORY ADAPTATION ON FRACTIONAL DUFFING (beta={beta})")
    print("=" * 78)

    # 1. Run AdaMem-FDE Adaptive Solver
    solver_adapt = AdaptiveSOEFDESolver(
        beta=beta,
        tol=tol,
        K_init=8,
        K_min=4,
        K_max=36,
        delta_K=4,
        prune_ratio=0.1,
        patience=4,
    )
    t0 = time.perf_counter()
    sol = solver_adapt.solve(duffing.rhs, z0, t_grid)
    runtime = (time.perf_counter() - t0) * 1000.0

    print(f"Simulation Horizon:        T = {T}, N = {num_steps} steps")
    print(f"Memory Tolerance:          eps_tol = {tol}")
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
        print(f"  Event {i+1:02d} at t={ev.t:5.2f} (step {ev.step_idx:3d}): "
              f"{ev.K_old} -> {ev.K_new} modes ({ev.event_type.upper()}), err={ev.error_est:.3e}")
    print("=" * 78)

    # Visualization: 3-panel publication figure
    t_np = t_grid.cpu().numpy()
    z_np = sol.z.cpu().numpy()
    modes_np = np.array(sol.modes_history)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Trajectory
    ax1.plot(t_np, z_np[:, 0], "b-", lw=1.8, label="x_1 (Displacement)")
    ax1.plot(t_np, z_np[:, 1], "r--", lw=1.5, label="x_2 (Velocity)")
    ax1.set_ylabel("State Variables", fontsize=10)
    ax1.set_title("Fractional Duffing Oscillator Trajectory", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Memory Mode Evolution K(t)
    ax2.step(t_np, modes_np, "g-", lw=2.0, where="post", label=f"Active Modes K(t) (Avg={sol.avg_modes:.1f})")
    for ev in expansions:
        ax2.axvline(ev.t, color="magenta", linestyle=":", alpha=0.6, label="Expand" if ev == expansions[0] else "")
    for ev in prunes:
        ax2.axvline(ev.t, color="teal", linestyle="--", alpha=0.6, label="Prune" if ev == prunes[0] else "")
    ax2.set_ylabel("Memory Modes K", fontsize=10)
    ax2.set_title("Dynamic Memory Representation K(t)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Phase portrait
    ax3.plot(z_np[:, 0], z_np[:, 1], "purple", lw=1.2)
    ax3.set_xlabel("Displacement x_1", fontsize=10)
    ax3.set_ylabel("Velocity x_2", fontsize=10)
    ax3.set_title("Phase Portrait (x_1 vs x_2)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "phase3_dynamic_memory_adaptation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Artifact Saved] Phase III adaptation figure saved to {plot_path}\n")


if __name__ == "__main__":
    run_phase3_experiment()
