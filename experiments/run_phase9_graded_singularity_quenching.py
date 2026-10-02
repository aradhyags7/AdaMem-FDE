"""
Phase IX: Graded Temporal Meshes & Singularity Quenching Benchmark.

Validates the mathematical elimination of the Caputo initial weak singularity error floor:
    \dot{z}(t) \sim t^{\beta - 1}, \quad \text{as } t \to 0^+

Under uniform grids (dt = T/N), fractional solvers suffer severe order reduction from O(N^-2)
to O(N^-beta). By deploying graded temporal meshes:
    t_n = T * (n / N)^r, \quad r = (2 - \beta) / \beta \ge 1
we restore the optimal second-order global convergence rate O(N^-2).

Outputs:
- results/phase9_graded_singularity_quenching.png
- results/phase9_graded_singularity_quenching.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scipy.special as sp
import torch
import matplotlib.pyplot as plt

from core.fractional.graded_mesh import generate_graded_mesh, optimal_grading_exponent
from core.fractional.mittag_leffler import mittag_leffler
from core.solvers.fixed_soe import FixedSOEFDESolver
from experiments.archive_utils import archive_previous_results


def run_singularity_quenching_experiment():
    print("=" * 78)
    print("PHASE IX: GRADED TEMPORAL MESHES & SINGULARITY QUENCHING BENCHMARK")
    print("=" * 78)

    # 1. Archive any existing results to preserve history
    target_files = [
        "phase9_graded_singularity_quenching.png",
        "phase9_graded_singularity_quenching.json",
    ]
    archive_previous_results("phase9_graded_singularity", target_files)
    os.makedirs("results", exist_ok=True)

    T = 1.0
    lam_scalar = 1.5
    betas = [0.5, 0.7, 0.85]
    N_values = [50, 100, 200, 400, 800]
    K_modes = 48  # Sufficient modes so time discretization error dominates

    def f(z, t):
        return -lam_scalar * z

    z0 = torch.tensor([1.0], dtype=torch.float64)

    results_data = {
        "T": T,
        "lam": lam_scalar,
        "betas": betas,
        "N_values": N_values,
        "K_modes": K_modes,
        "runs": {},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("#0f172a")

    for idx, beta in enumerate(betas):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")

        r_opt = optimal_grading_exponent(beta)
        print(f"\nEvaluating beta = {beta:.2f} (optimal grading exponent r = {r_opt:.3f}):")

        uniform_errors = []
        graded_errors = []

        for N in N_values:
            # 1. Uniform Mesh
            t_unif = torch.linspace(0.0, T, N + 1, dtype=torch.float64)
            sol_u = FixedSOEFDESolver(beta=beta, num_modes=K_modes).solve(f, z0, t_unif).squeeze(-1)
            exact_u = torch.tensor(
                mittag_leffler(beta, 1.0, -lam_scalar * (t_unif.numpy() ** beta)),
                dtype=torch.float64,
            )
            err_u = float(torch.max(torch.abs(sol_u - exact_u)).item())
            uniform_errors.append(err_u)

            # 2. Graded Mesh
            t_grad = generate_graded_mesh(T=T, N=N, beta=beta, r=r_opt, dtype=torch.float64)
            sol_g = FixedSOEFDESolver(beta=beta, num_modes=K_modes).solve(f, z0, t_grad).squeeze(-1)
            exact_g = torch.tensor(
                mittag_leffler(beta, 1.0, -lam_scalar * (t_grad.numpy() ** beta)),
                dtype=torch.float64,
            )
            err_g = float(torch.max(torch.abs(sol_g - exact_g)).item())
            graded_errors.append(err_g)

            ratio = err_u / max(err_g, 1e-12)
            print(f"  N = {N:4d} | Uniform Error: {err_u:.3e} | Graded Error: {err_g:.3e} | Improvement: {ratio:.2f}x")

        # Empirical convergence slopes (log-log)
        p_u = np.polyfit(np.log10(N_values), np.log10(uniform_errors), 1)[0]
        p_g = np.polyfit(np.log10(N_values), np.log10(graded_errors), 1)[0]

        results_data["runs"][str(beta)] = {
            "r_opt": r_opt,
            "uniform_errors": uniform_errors,
            "graded_errors": graded_errors,
            "slope_uniform": float(p_u),
            "slope_graded": float(p_g),
        }

        # Plot curves
        ax.loglog(N_values, uniform_errors, "o--", color="#ef4444", lw=2.2, ms=7, label=f"Uniform (Slope: {p_u:.2f})")
        ax.loglog(N_values, graded_errors, "s-", color="#10b981", lw=2.5, ms=7, label=f"Graded r={r_opt:.2f} (Slope: {p_g:.2f})")

        # Reference slopes
        N_arr = np.array(N_values, dtype=float)
        ref_o1 = uniform_errors[0] * (N_arr[0] / N_arr) ** beta
        ref_o2 = graded_errors[0] * (N_arr[0] / N_arr) ** 2.0
        ax.loglog(N_values, ref_o1, ":", color="#94a3b8", alpha=0.6, label=f"Theory O(N^{{-{beta}}})")
        ax.loglog(N_values, ref_o2, "-.", color="#38bdf8", alpha=0.6, label="Theory O(N^-2)")

        ax.set_title(f"Fractional Order $\\beta = {beta}$ ($r = {r_opt:.2f}$)", color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mesh Intervals $N$", color="#cbd5e1", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"Maximum Global Error $\|z - z_{exact}\|_\infty$", color="#cbd5e1", fontsize=11)

        ax.grid(True, which="both", ls=":", color="#334155", alpha=0.7)
        ax.tick_params(colors="#cbd5e1")
        leg = ax.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
        for text in leg.get_texts():
            text.set_color("white")

    plt.suptitle("Caputo Weak Singularity Quenching via Graded Temporal Meshes", color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = os.path.join("results", "phase9_graded_singularity_quenching.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    json_path = os.path.join("results", "phase9_graded_singularity_quenching.json")
    with open(json_path, "w") as fp:
        json.dump(results_data, fp, indent=2)

    print(f"\n[Saved] Figure: {plot_path}")
    print(f"[Saved] Dataset: {json_path}")
    print("=" * 78)


if __name__ == "__main__":
    run_singularity_quenching_experiment()
