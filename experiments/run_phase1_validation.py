"""
Phase I & II Experiment: Fractional Solver Validation on Analytical Mittag-Leffler Decay.

Compares:
1. Analytical Ground Truth: x(t) = x_0 E_\beta(-\lambda t^\beta)
2. Full-History Adams-Bashforth-Moulton (Baseline 1, O(N^2))
3. Fixed-Order SOE (Baseline 2, K = 8, 16, 24)
4. AdaMem-FDE (Proposed Error-Adaptive Solver)
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.systems import MittagLefflerDecay
from core.solvers.full_history import FullHistoryFDESolver
from core.solvers.fixed_soe import FixedSOEFDESolver
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver


def run_phase1_experiment(
    beta: float = 0.7,
    decay_rate: float = 1.0,
    T: float = 5.0,
    num_steps: int = 500,
    save_dir: str = "results",
):
    os.makedirs(save_dir, exist_ok=True)
    from experiments.archive_utils import archive_previous_results
    archive_previous_results("phase1_validation", ["phase1_mittag_leffler_benchmark.png", "phase1_mittag_leffler_data.json"], save_dir=save_dir)

    t_grid = torch.linspace(0.0, T, num_steps + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)

    system = MittagLefflerDecay(beta=beta, decay_rate=decay_rate, x0=1.0)
    z_exact = system.exact_solution(t_grid)
    exact_norm = torch.norm(z_exact).item()

    results = []

    print("\n" + "=" * 78)
    print(f"  PHASE I & II: MITTAG-LEFFLER VALIDATION (beta={beta}, T={T}, N={num_steps})")
    print("=" * 78)
    print(f"{'Method':<24} | {'Forward Err':<12} | {'Time (ms)':<10} | {'Modes (Avg/Max)':<16}")
    print("-" * 78)

    # 1. Full-History ABM Solver
    t0 = time.perf_counter()
    solver_full = FullHistoryFDESolver(beta=beta)
    z_full = solver_full.solve(system.rhs, z0, t_grid).squeeze(-1)
    t_full = (time.perf_counter() - t0) * 1000.0
    err_full = torch.norm(z_full - z_exact).item() / exact_norm
    results.append(("Full-History (O(N^2))", z_full, err_full, t_full, num_steps, num_steps))
    print(f"{'Full-History O(N^2)':<24} | {err_full:<12.4e} | {t_full:<10.2f} | {num_steps}/{num_steps}")

    # 2. Fixed SOE Solvers (K = 8, 16, 24)
    for K in [8, 16, 24]:
        t0 = time.perf_counter()
        solver_fixed = FixedSOEFDESolver(beta=beta, num_modes=K)
        z_fixed = solver_fixed.solve(system.rhs, z0, t_grid).squeeze(-1)
        t_fixed = (time.perf_counter() - t0) * 1000.0
        err_fixed = torch.norm(z_fixed - z_exact).item() / exact_norm
        results.append((f"Fixed SOE (K={K})", z_fixed, err_fixed, t_fixed, K, K))
        print(f"{f'Fixed SOE (K={K})':<24} | {err_fixed:<12.4e} | {t_fixed:<10.2f} | {K}/{K}")

    # 3. AdaMem-FDE Adaptive Solvers (tol = 1e-3, 1e-4)
    for tol in [1e-3, 1e-4]:
        t0 = time.perf_counter()
        solver_adapt = AdaptiveSOEFDESolver(beta=beta, tol=tol, K_init=8, K_min=4, K_max=32)
        sol_adapt = solver_adapt.solve(system.rhs, z0, t_grid)
        z_adapt = sol_adapt.z.squeeze(-1)
        t_adapt = (time.perf_counter() - t0) * 1000.0
        err_adapt = torch.norm(z_adapt - z_exact).item() / exact_norm
        avg_k = sol_adapt.avg_modes
        max_k = sol_adapt.max_modes
        results.append((f"AdaMem (tol={tol})", z_adapt, err_adapt, t_adapt, avg_k, max_k))
        print(f"{f'AdaMem (tol={tol})':<24} | {err_adapt:<12.4e} | {t_adapt:<10.2f} | {avg_k:.1f}/{max_k}")

    print("=" * 78)

    # Visualization
    t_np = t_grid.cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Trajectories
    ax1.plot(t_np, z_exact.cpu().numpy(), "k-", lw=2.5, label="Analytical (Mittag-Leffler)")
    for name, z_arr, err, _, _, _ in results:
        ax1.plot(t_np, z_arr.cpu().numpy(), "--", lw=1.5, label=f"{name} (E={err:.2e})")
    ax1.set_xlabel("Time t", fontsize=11)
    ax1.set_ylabel("State x(t)", fontsize=11)
    ax1.set_title("Forward Solution Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    # Pointwise Absolute Error
    for name, z_arr, _, _, _, _ in results:
        pt_err = np.abs(z_arr.cpu().numpy() - z_exact.cpu().numpy())
        ax2.semilogy(t_np[1:], pt_err[1:], label=name, lw=1.5)
    ax2.set_xlabel("Time t", fontsize=11)
    ax2.set_ylabel("Absolute Error |x_num(t) - x_exact(t)|", fontsize=11)
    ax2.set_title("Pointwise Error vs Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "phase1_mittag_leffler_benchmark.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Save structured benchmark data for research paper reporting
    import json
    data_record = []
    for name, _, err, runtime, avg_k, max_k in results:
        data_record.append({
            "method": name,
            "forward_error": float(err),
            "runtime_ms": float(runtime),
            "avg_modes": float(avg_k),
            "max_modes": int(max_k),
        })
    json_path = os.path.join(save_dir, "phase1_mittag_leffler_data.json")
    with open(json_path, "w") as f:
        json.dump(data_record, f, indent=2)

    print(f"\n[Artifact Saved] Benchmark figure saved to {plot_path}")
    print(f"[Artifact Saved] Benchmark data saved to {json_path}\n")


if __name__ == "__main__":
    run_phase1_experiment()
