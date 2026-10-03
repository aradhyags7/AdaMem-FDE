"""
Phase X: Incommensurate Multi-Order Fractional Dynamics Benchmark.

Validates incommensurate fractional systems where physical state variables exhibit
heterogeneous anomalous memory retention:
    {}^C D_0^{\vec{\beta}} z(t) = f(z(t), t) \iff
    {}^C D_0^{\beta_i} z_i(t) = f_i(z(t), t), \quad \beta_i \in (0, 1), \quad i = 1, \dots, d

Evaluates a coupled multi-scale fractional FitzHugh-Nagumo neural oscillator:
    ^C D^{\beta_1} v = v - v^3/3 - w + I_ext,   (fast membrane voltage, beta_1 = 0.90)
    ^C D^{\beta_2} w = eps * (v + a - b * w),   (slow adaptation variable, beta_2 = 0.60)

Demonstrates:
1. Commensurate scalar beta models fail to capture multi-timescale relaxation dynamics.
2. Incommensurate multi-order models recover both ground-truth exponents simultaneously
   using exact analytical vector sensitivities \nabla_{\vec{\beta}} \mathcal{L}.

Outputs:
- results/phase10_incommensurate_multi_order.png
- results/phase10_incommensurate_multi_order.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from core.solvers.incommensurate_soe import IncommensurateSOEFDESolver, incommensurate_adamem_integrate
from models.neural_fde import NeuralFDE, VectorFieldNetwork
from experiments.archive_utils import archive_previous_results


def run_incommensurate_experiment():
    print("=" * 78)
    print("PHASE X: INCOMMENSURATE MULTI-ORDER FRACTIONAL DYNAMICS BENCHMARK")
    print("=" * 78)

    # 1. Archive previous results if any
    target_files = [
        "phase10_incommensurate_multi_order.png",
        "phase10_incommensurate_multi_order.json",
    ]
    archive_previous_results("phase10_incommensurate", target_files)
    os.makedirs("results", exist_ok=True)

    # System parameters: Fractional FitzHugh-Nagumo
    true_beta = torch.tensor([0.90, 0.60], dtype=torch.float32)
    eps = 0.08
    a = 0.7
    b = 0.8
    I_ext = 0.5
    T = 20.0
    N = 200
    t_grid = torch.linspace(0.0, T, N + 1, dtype=torch.float32)
    z0 = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)  # (1, 2)

    def f_fhn(z, t):
        # z: (..., 2)
        v = z[..., 0:1]
        w = z[..., 1:2]
        dv = v - (v ** 3) / 3.0 - w + I_ext
        dw = eps * (v + a - b * w)
        return torch.cat([dv, dw], dim=-1)

    print(f"Generating ground-truth trajectory with true beta = [{true_beta[0]:.2f}, {true_beta[1]:.2f}]...")
    solver_true = IncommensurateSOEFDESolver(beta=true_beta, num_modes=32)
    z_true = solver_true.solve(f_fhn, z0, t_grid).detach()  # (N+1, 1, 2)

    # 2. Evaluate Commensurate Baselines (scalar beta = 0.75, 0.90, 0.60)
    print("\nEvaluating Commensurate Baselines...")
    commensurate_betas = [0.60, 0.75, 0.90]
    comm_trajs = {}
    comm_errors = {}
    for cb in commensurate_betas:
        s_comm = IncommensurateSOEFDESolver(beta=torch.tensor([cb, cb]), num_modes=32)
        z_comm = s_comm.solve(f_fhn, z0, t_grid).detach()
        err = float(torch.mean((z_comm - z_true) ** 2).item())
        key = f"{cb:.2f}"
        comm_trajs[key] = z_comm.squeeze(1).numpy()
        comm_errors[key] = err
        print(f"  Commensurate beta = {cb:.2f} | MSE: {err:.4e}")


    # 3. Vector Beta Optimization using Incommensurate Adjoint Sensitivities
    print("\nOptimizing Incommensurate Multi-Order Beta starting from [0.50, 0.50]...")
    init_beta = [0.50, 0.50]

    # Simple wrapper module for f_fhn to make it compatible with NeuralFDE
    class FHNModule(nn.Module):
        def forward(self, z, t):
            return f_fhn(z, t)

    fhn_net = FHNModule()
    model = NeuralFDE(
        vector_field=fhn_net,
        beta=init_beta,
        learnable_beta=True,
        K_init=24,
    )

    optimizer = torch.optim.Adam([model._raw_beta], lr=0.08)
    history_beta = []
    history_loss = []

    num_epochs = 40
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        current_b = model.get_beta_value()
        history_beta.append([float(current_b[0]), float(current_b[1])])

        pred = model(z0, t_grid, method="adamem")
        loss = torch.mean((pred - z_true) ** 2)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        history_loss.append(loss_val)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} | Loss: {loss_val:.4e} | Recovered Beta: [{current_b[0]:.4f}, {current_b[1]:.4f}]")

    final_beta = model.get_beta_value()
    print(f"\nFinal Recovered Beta: [{final_beta[0]:.4f}, {final_beta[1]:.4f}] vs True [{true_beta[0]:.4f}, {true_beta[1]:.4f}]")

    # 4. Multi-Panel Figure Generation
    fig = plt.figure(figsize=(18, 5.5), facecolor="white")

    # Panel (a): Voltage Phase Portrait & Trajectory
    ax1 = fig.add_subplot(1, 3, 1, facecolor="white")
    t_np = t_grid.numpy()
    z_true_np = z_true.squeeze(1).numpy()
    ax1.plot(t_np, z_true_np[:, 0], color="#0284c7", lw=2.5, label=r"True $v(t)$ ($\beta_1=0.90$)")
    ax1.plot(t_np, z_true_np[:, 1], color="#d97706", lw=2.5, label=r"True $w(t)$ ($\beta_2=0.60$)")
    ax1.plot(t_np, comm_trajs["0.75"][:, 0], "--", color="#dc2626", lw=1.8, alpha=0.8, label=r"Commensurate $\bar{\beta}=0.75$")
    ax1.set_title("Multi-Order FitzHugh-Nagumo Dynamics", color="#0f172a", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Time $t$", color="#0f172a", fontsize=11)
    ax1.set_ylabel("State Magnitude", color="#0f172a", fontsize=11)
    ax1.grid(True, ls=":", color="#cbd5e1", alpha=0.7)
    ax1.tick_params(colors="#1e293b")
    ax1.legend(facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # Panel (b): Vector Beta Optimization Trajectory in Parameter Space
    ax2 = fig.add_subplot(1, 3, 2, facecolor="white")
    h_beta = np.array(history_beta)
    ax2.plot(h_beta[:, 0], h_beta[:, 1], "o-", color="#059669", lw=2.2, ms=5, label=r"Trajectory $(\beta_1^{(k)}, \beta_2^{(k)})$")
    ax2.plot([h_beta[0, 0]], [h_beta[0, 1]], "s", color="#e11d48", ms=9, label=r"Initial $(0.50, 0.50)$")
    ax2.plot([true_beta[0]], [true_beta[1]], "*", color="#0284c7", ms=14, label=r"True Target $(0.90, 0.60)$")
    ax2.set_xlim(0.45, 0.95)
    ax2.set_ylim(0.45, 0.70)
    ax2.set_title(r"Parameter Space Convergence $\vec{\beta} \in \mathbb{R}^2$", color="#0f172a", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel(r"Voltage Exponent $\beta_1$", color="#0f172a", fontsize=11)
    ax2.set_ylabel(r"Recovery Exponent $\beta_2$", color="#0f172a", fontsize=11)
    ax2.grid(True, ls=":", color="#cbd5e1", alpha=0.7)
    ax2.tick_params(colors="#1e293b")
    ax2.legend(facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # Panel (c): Optimization Loss & Commensurate Distortion Comparison
    ax3 = fig.add_subplot(1, 3, 3, facecolor="white")
    ax3.semilogy(range(1, num_epochs + 1), history_loss, color="#7c3aed", lw=2.5, label="Incommensurate AdaMem Loss")
    ax3.axhline(comm_errors["0.75"], ls="--", color="#dc2626", lw=2.0, label=r"Commensurate $\bar{\beta}=0.75$ Floor")
    ax3.axhline(comm_errors["0.90"], ls=":", color="#ea580c", lw=2.0, label=r"Commensurate $\bar{\beta}=0.90$ Floor")
    ax3.set_title("Distortion Dissolution vs Commensurate Floors", color="#0f172a", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Adjoint Optimization Epoch", color="#0f172a", fontsize=11)
    ax3.set_ylabel("Trajectory Mean Squared Error", color="#0f172a", fontsize=11)
    ax3.grid(True, which="both", ls=":", color="#cbd5e1", alpha=0.7)
    ax3.tick_params(colors="#1e293b")
    ax3.legend(facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    plt.suptitle("Incommensurate Multi-Order Fractional Dynamics & Vector Sensitivity Identification", color="#0f172a", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = os.path.join("results", "phase10_incommensurate_multi_order.png")
    paper_fig_path = os.path.join("paper", "figures", "phase10_incommensurate_multi_order.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(paper_fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    results_data = {
        "true_beta": [float(true_beta[0]), float(true_beta[1])],
        "final_recovered_beta": [float(final_beta[0]), float(final_beta[1])],
        "commensurate_errors": comm_errors,
        "history_loss": history_loss,
        "history_beta": history_beta,
    }

    json_path = os.path.join("results", "phase10_incommensurate_multi_order.json")
    with open(json_path, "w") as fp:
        json.dump(results_data, fp, indent=2)

    print(f"\n[Saved] Figure: {plot_path}")
    print(f"[Saved] Dataset: {json_path}")
    print("=" * 78)


if __name__ == "__main__":
    run_incommensurate_experiment()
