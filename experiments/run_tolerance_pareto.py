"""
Phase VII / RQ6 Experiment: Multi-Tolerance Pareto Frontier & Sensitivity Analysis.

Investigates the multi-dimensional trade-off:
    Accuracy E_z <---> Memory Complexity \\bar{K} <---> Wall-Clock Runtime <---> Gradient Error E_g

Sweeps tolerance \\epsilon_{tol} \\in [10^{-2}, 10^{-6}] and compares against fixed-order SOE baselines
to empirically characterize the Pareto optimality of AdaMem-FDE.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.systems import MittagLefflerDecay, FractionalDuffing
from core.solvers.adaptive_soe import AdaptiveSOEFDESolver
from core.solvers.fixed_soe import FixedSOEFDESolver
from core.adjoint.adamem_adjoint import adamem_integrate


def run_tolerance_sweep_mittag_leffler(
    beta: float = 0.7,
    T: float = 4.0,
    N: int = 400,
    tolerances: list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    fixed_modes: list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
):
    """
    Evaluates Pareto trade-off on analytical Mittag-Leffler decay where ground truth is known.
    """
    t_grid = torch.linspace(0.0, T, N + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)
    system = MittagLefflerDecay(beta=beta, decay_rate=1.0, x0=1.0)
    z_exact = system.exact_solution(t_grid)
    exact_norm = torch.norm(z_exact).item()

    # 1. Sweep Fixed SOE Baselines
    fixed_results = []
    for K in fixed_modes:
        solver = FixedSOEFDESolver(beta=beta, num_modes=K)
        # Warmup
        _ = solver.solve(system.rhs, z0, t_grid)
        # Benchmark timing
        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            z_num = solver.solve(system.rhs, z0, t_grid).squeeze(-1)
        runtime = (time.perf_counter() - t0) / runs * 1000.0

        err = torch.norm(z_num - z_exact).item() / exact_norm
        fixed_results.append({
            "K": K,
            "error": err,
            "runtime_ms": runtime,
        })

    # 2. Sweep AdaMem-FDE Across Tolerances
    adaptive_results = []
    for tol in tolerances:
        solver = AdaptiveSOEFDESolver(
            beta=beta,
            tol=tol,
            K_init=6,
            K_min=4,
            K_max=44,
            delta_K=4,
            prune_ratio=0.08,
            patience=3,
        )
        # Warmup
        _ = solver.solve(system.rhs, z0, t_grid)
        # Benchmark timing
        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            sol = solver.solve(system.rhs, z0, t_grid)
        runtime = (time.perf_counter() - t0) / runs * 1000.0

        err = torch.norm(sol.z.squeeze(-1) - z_exact).item() / exact_norm

        adaptive_results.append({
            "tol": tol,
            "error": err,
            "avg_modes": sol.avg_modes,
            "max_modes": sol.max_modes,
            "n_adapt": sol.num_adaptations,
            "runtime_ms": runtime,
        })

    return fixed_results, adaptive_results


def run_gradient_error_sweep(
    beta: float = 0.8,
    T: float = 1.0,
    N: int = 40,
    tolerances: list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
):
    """
    Evaluates gradient fidelity E_g across memory tolerances against finite difference.
    """
    t_grid = torch.linspace(0.0, T, N + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0], dtype=torch.float64)
    z_target = torch.exp(-0.8 * t_grid).unsqueeze(-1)

    class SimpleParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        def forward(self, z, t):
            return -self.theta * z

    grad_results = []

    # High-precision central finite difference
    h = 1e-6
    with torch.no_grad():
        m_ref = SimpleParamModel()
        m_ref.theta.data.fill_(1.0 + h)
        zp = adamem_integrate(m_ref, z0, t_grid, beta=beta, tol=1e-5, K_init=8, K_min=4, K_max=32, use_adjoint_jump=True)
        lp = 0.5 * torch.sum((zp - z_target) ** 2).item()

        m_ref.theta.data.fill_(1.0 - h)
        zm = adamem_integrate(m_ref, z0, t_grid, beta=beta, tol=1e-5, K_init=8, K_min=4, K_max=32, use_adjoint_jump=True)
        lm = 0.5 * torch.sum((zm - z_target) ** 2).item()
        g_ref = (lp - lm) / (2.0 * h)

    for tol in tolerances:
        m = SimpleParamModel()
        z_traj = adamem_integrate(
            m, z0, t_grid, beta=beta, tol=tol, K_init=8, K_min=4, K_max=32,
            use_adjoint_jump=True, parameters=tuple(m.parameters())
        )
        loss = 0.5 * torch.sum((z_traj - z_target) ** 2)
        loss.backward()
        g_adj = m.theta.grad.item()
        e_g = abs(g_adj - g_ref) / max(abs(g_ref), 1e-7)

        grad_results.append({
            "tol": tol,
            "g_adj": g_adj,
            "g_ref": g_ref,
            "grad_err": e_g,
        })

    return grad_results


def run_tolerance_pareto_experiment(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 84)
    print("  PARETO FRONTIER & TOLERANCE SENSITIVITY EXPERIMENT (AdaMem-FDE vs Fixed SOE)")
    print("=" * 84)

    tolerances = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    fixed_modes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

    fixed_res, adapt_res = run_tolerance_sweep_mittag_leffler(
        beta=0.7, T=4.0, N=400, tolerances=tolerances, fixed_modes=fixed_modes
    )
    grad_res = run_gradient_error_sweep(
        beta=0.8, T=1.0, N=40, tolerances=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    )

    # Print Summary Table
    print(f"\n{'Tolerance':<12} | {'Forward Err E_z':<18} | {'Avg Modes K':<14} | {'Max Modes':<12} | {'Adaptations':<12} | {'Time (ms)':<10}")
    print("-" * 84)
    for r in adapt_res:
        print(f"{r['tol']:<12.1e} | {r['error']:<18.4e} | {r['avg_modes']:<14.2f} | {r['max_modes']:<12d} | {r['n_adapt']:<12d} | {r['runtime_ms']:<10.2f}")
    print("=" * 84)

    # Publication-Grade 4-Panel Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Pareto Frontier: Error vs Modes
    fix_k = [r["K"] for r in fixed_res]
    fix_err = [r["error"] for r in fixed_res]
    ada_k = [r["avg_modes"] for r in adapt_res]
    ada_err = [r["error"] for r in adapt_res]

    ax1.semilogy(fix_k, fix_err, "r--s", lw=2, ms=6, label="Fixed-Order SOE Baseline")
    ax1.semilogy(ada_k, ada_err, "b-o", lw=2.5, ms=7, label="AdaMem-FDE (Adaptive)")
    ax1.set_xlabel(r"Memory Complexity: Average Modes $\bar{K}$", fontsize=11)
    ax1.set_ylabel(r"Forward Relative Error $E_z$", fontsize=11)
    ax1.set_title(r"(a) Accuracy vs. Memory Modes $\bar{K}$", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: Error vs Runtime
    fix_t = [r["runtime_ms"] for r in fixed_res]
    ada_t = [r["runtime_ms"] for r in adapt_res]

    ax2.loglog(fix_t, fix_err, "r--s", lw=2, ms=6, label="Fixed-Order SOE Baseline")
    ax2.loglog(ada_t, ada_err, "b-o", lw=2.5, ms=7, label="AdaMem-FDE (Adaptive)")
    ax2.set_xlabel("Runtime (ms, log scale)", fontsize=11)
    ax2.set_ylabel(r"Forward Relative Error $E_z$", fontsize=11)
    ax2.set_title(r"(b) Accuracy vs. Computational Runtime (Python)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper right")

    # Panel 3: Active Modes vs Tolerance
    ada_tol = [r["tol"] for r in adapt_res]
    ada_max = [r["max_modes"] for r in adapt_res]

    ax3.semilogx(ada_tol, ada_k, "b-o", lw=2.2, ms=7, label=r"Average Modes $\bar{K}$")
    ax3.semilogx(ada_tol, ada_max, "m--^", lw=1.8, ms=6, label=r"Maximum Modes $K_{\max}$")
    ax3.set_xlabel(r"Prescribed Memory Tolerance $\epsilon_{\mathrm{tol}}$", fontsize=11)
    ax3.set_ylabel("Active Memory Modes", fontsize=11)
    ax3.set_title(r"(c) Dynamic Mode Allocation vs. Tolerance", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc="upper right")

    # Panel 4: Gradient Error vs Tolerance
    g_tol = [r["tol"] for r in grad_res]
    g_err = [r["grad_err"] for r in grad_res]

    ax4.semilogx(g_tol, g_err, "teal", marker="D", lw=2.0, ms=7, label=r"Adjoint Error $E_g$")
    ax4.set_xlabel(r"Prescribed Memory Tolerance $\epsilon_{\mathrm{tol}}$", fontsize=11)
    ax4.set_ylabel(r"Relative Gradient Error $E_g$", fontsize=11)
    ax4.set_title(r"(d) Gradient Fidelity vs. Memory Tolerance", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "pareto_frontier_analysis.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\n[Artifact Saved] Pareto frontier visualization saved to {plot_path}\n")


if __name__ == "__main__":
    run_tolerance_pareto_experiment()
