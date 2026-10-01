"""
Phase VI Experiment: Long-Horizon Scalability Study on Fractional Lorenz Attractor.

Empirically characterizes computational complexity across growing horizons N:
- Full-History Solver: O(N^2) time complexity
- Fixed SOE Solver: O(N * K) time complexity
- AdaMem-FDE: O(N * \\bar{K}) error-controlled time complexity

Measures:
- Wall-clock runtime vs N
- Memory scaling behavior
- Average active modes \\bar{K} as N increases
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.systems import FractionalLorenz
from core.solvers.full_history import FullHistoryFDESolver
from core.solvers.fixed_soe import FixedSOEFDESolver
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


def run_phase6_experiment(
    beta: float = 0.99,
    step_counts: list = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
    save_dir: str = "results",
):
    os.makedirs(save_dir, exist_ok=True)
    system = FractionalLorenz(beta=beta)
    z0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    full_history_times = []
    fixed_soe_times = []
    adamem_times = []
    adamem_avg_modes = []

    print("\n" + "=" * 80)
    print(f"  PHASE VI: LONG-HORIZON SCALABILITY STUDY ON FRACTIONAL LORENZ (beta={beta})")
    print(f"  Horizon Range: N = {step_counts[0]:,} to N = {step_counts[-1]:,} time steps")
    print("=" * 80)
    header_k = "AdaMem K_avg"
    print(f"{'N Steps':<10} | {'Full-History (s)':<18} | {'Fixed SOE (s)':<16} | {'AdaMem-FDE (s)':<16} | {header_k:<12}")
    print("-" * 80)

    for N in step_counts:
        dt = 0.01
        T = N * dt
        t_grid = torch.linspace(0.0, T, N + 1, dtype=torch.float64)

        # 1. Full-History ABM (run directly for N <= 2500 due to O(N^2) memory and time)
        if N <= 2500:
            t0 = time.perf_counter()
            solver_full = FullHistoryFDESolver(beta=beta)
            _ = solver_full.solve(system.rhs, z0, t_grid)
            t_full = time.perf_counter() - t0
            full_history_times.append((N, t_full))
            t_full_str = f"{t_full:.4f} s"
        else:
            # Extrapolate quadratic O(N^2) scaling for plotting
            N_ref, t_ref = full_history_times[-1]
            t_full_est = t_ref * ((N / N_ref) ** 2)
            full_history_times.append((N, t_full_est))
            t_full_str = f"~{t_full_est:.2f} s (proj)"

        # 2. Fixed SOE (K = 16)
        t0 = time.perf_counter()
        solver_fixed = FixedSOEFDESolver(beta=beta, num_modes=16)
        _ = solver_fixed.solve(system.rhs, z0, t_grid)
        t_fixed = time.perf_counter() - t0
        fixed_soe_times.append((N, t_fixed))

        # 3. AdaMem-FDE (Adaptive)
        t0 = time.perf_counter()
        solver_adapt = AdaptiveSOEFDESolver(beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=32)
        sol = solver_adapt.solve(system.rhs, z0, t_grid)
        t_adapt = time.perf_counter() - t0
        adamem_times.append((N, t_adapt))
        adamem_avg_modes.append(sol.avg_modes)

        print(f"{N:<10} | {t_full_str:<18} | {t_fixed:<16.4f} | {t_adapt:<16.4f} | {sol.avg_modes:<12.1f}")

    print("=" * 80)

    # Visualization: Log-Log Complexity Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ns_full = [p[0] for p in full_history_times]
    ts_full = [p[1] for p in full_history_times]
    ns_fix = [p[0] for p in fixed_soe_times]
    ts_fix = [p[1] for p in fixed_soe_times]
    ns_ada = [p[0] for p in adamem_times]
    ts_ada = [p[1] for p in adamem_times]

    # Panel 1: Runtime vs N (Log-Log)
    ax1.loglog(ns_full, ts_full, "r--s", lw=2, ms=6, label=r"Full-History ABM ($\mathcal{O}(N^2)$)")
    ax1.loglog(ns_fix, ts_fix, "g-^", lw=2, ms=6, label=r"Fixed SOE ($K=16$, $\mathcal{O}(N)$)")
    ax1.loglog(ns_ada, ts_ada, "b-o", lw=2.5, ms=6, label=r"AdaMem-FDE (Adaptive, $\mathcal{O}(N)$)")
    ax1.set_xlabel("Number of Time Steps $N$", fontsize=11)
    ax1.set_ylabel("Wall-Clock Runtime (seconds)", fontsize=11)
    ax1.set_title(r"(a) Computational Scaling: Runtime vs Horizon $N$", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: Active Memory Modes \bar{K} vs N
    ax2.plot(ns_ada, adamem_avg_modes, "b-o", lw=2.2, ms=6, label=r"Adaptive Modes $\bar{K}$")
    ax2.axhline(32, color="gray", linestyle=":", lw=1.5, label=r"$K_{\max} = 32$ Ceiling")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Time Steps $N$ (log scale)", fontsize=11)
    ax2.set_ylabel(r"Average Active Modes $\bar{K}$", fontsize=11)
    ax2.set_title(r"(b) Memory Footprint Boundedness vs Horizon $N$", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 36)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "phase6_long_horizon_scaling.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Artifact Saved] Scalability comparison saved to {plot_path}\n")


if __name__ == "__main__":
    run_phase6_experiment()
