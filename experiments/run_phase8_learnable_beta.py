"""
Phase VIII Experiment: Joint Discovery of Fractional Order Beta and Dynamical Parameters.

Demonstrates:
1. Joint Optimization of vector field parameters \theta and nonlocal memory order \beta
   via the analytical adjoint sensitivity:
       \frac{\partial z}{\partial \beta} = -\frac{\psi(\beta)}{\Gamma(\beta)} \sum w_k m_k 
                                            + \frac{1}{\Gamma(\beta)} \sum \frac{\partial w_k}{\partial \beta} m_k
2. Comparison against:
   - Fixed Misspecified Baseline (\beta = 0.50 frozen)
   - Fixed Oracle Reference (\beta* = 0.75 known a priori)
3. Quantitative parameter recovery and dynamic memory mode tracking.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.adjoint.adamem_adjoint import adamem_integrate
from models.neural_fde import NeuralFDE


class TrueFractionalOscillator(nn.Module):
    """
    Ground truth damped fractional oscillator:
        {}^C D_t^\beta x_1 = x_2
        {}^C D_t^\beta x_2 = -\omega^2 x_1 - \mu x_2
    with true \beta* = 0.75, \omega^2 = 1.50, \mu = 0.50.
    """
    def __init__(self, omega2: float = 1.50, mu: float = 0.50):
        super().__init__()
        self.omega2 = omega2
        self.mu = mu

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        x1 = z[..., 0]
        x2 = z[..., 1]
        dx1 = x2
        dx2 = -self.omega2 * x1 - self.mu * x2
        return torch.stack([dx1, dx2], dim=-1)


class ParameterizedOscillator(nn.Module):
    """
    Trainable parameterized oscillator with initial parameter guesses:
        \omega^2 = 0.80 (target 1.50)
        \mu = 0.20 (target 0.50)
    """
    def __init__(self, init_omega2: float = 0.80, init_mu: float = 0.20):
        super().__init__()
        self.omega2 = nn.Parameter(torch.tensor(init_omega2, dtype=torch.float64))
        self.mu = nn.Parameter(torch.tensor(init_mu, dtype=torch.float64))

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        x1 = z[..., 0]
        x2 = z[..., 1]
        dx1 = x2
        dx2 = -self.omega2 * x1 - self.mu * x2
        return torch.stack([dx1, dx2], dim=-1)


def generate_ground_truth(true_beta: float = 0.75, T: float = 3.0, num_steps: int = 60):
    """Generates ground truth trajectory using high-accuracy AdaMem forward solver."""
    t_grid = torch.linspace(0.0, T, num_steps + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    system = TrueFractionalOscillator(omega2=1.50, mu=0.50)
    with torch.no_grad():
        z_true = adamem_integrate(
            system, z0, t_grid, beta=true_beta, tol=1e-4, K_init=12, K_min=4, K_max=32
        )
    return t_grid, z0, z_true


def run_phase8_experiment(save_dir: str = "results", epochs: int = 75):
    os.makedirs(save_dir, exist_ok=True)
    true_beta = 0.75
    init_beta = 0.50

    print("\n" + "=" * 82)
    print("  PHASE VIII: LEARNABLE FRACTIONAL ORDER (BETA) JOINT DISCOVERY BENCHMARK")
    print(f"  Target System: Damped Fractional Oscillator (True beta* = {true_beta:.2f})")
    print(f"  Initialization: beta_0 = {init_beta:.2f} (Misspecified by Delta beta = {init_beta - true_beta:+.2f})")
    print(f"  Training Schedule: {epochs} epochs with Cosine Annealing learning rate")
    print("=" * 82)

    t_grid, z0, z_true = generate_ground_truth(true_beta=true_beta, T=3.0, num_steps=60)

    # ---------------------------------------------------------
    # 1. Proposed Model: Joint Optimization of (\theta, \beta)
    # ---------------------------------------------------------
    print("\n[Model 1] Training Proposed Joint AdaMem-FDE (Learnable beta + Vector Field)...")
    field_joint = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
    model_joint = NeuralFDE(
        vector_field=field_joint,
        beta=init_beta,
        learnable_beta=True,
        default_tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
    )
    param_groups = [
        {"params": [model_joint._raw_beta], "lr": 0.12},
        {"params": field_joint.parameters(), "lr": 0.04},
    ]
    optimizer_joint = torch.optim.Adam(param_groups)
    scheduler_joint = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_joint, T_max=epochs, eta_min=5e-3)

    loss_history_joint = []
    beta_history_joint = []
    omega_history_joint = []
    mu_history_joint = []

    t0_joint = time.perf_counter()
    for ep in range(epochs):
        optimizer_joint.zero_grad()
        pred_joint = model_joint(z0, t_grid, method="adamem", tol=1e-3)
        loss = torch.mean((pred_joint - z_true) ** 2)
        loss.backward()
        optimizer_joint.step()
        scheduler_joint.step()

        curr_beta = model_joint.get_beta_value()
        curr_omega = field_joint.omega2.item()
        curr_mu = field_joint.mu.item()

        loss_history_joint.append(loss.item())
        beta_history_joint.append(curr_beta)
        omega_history_joint.append(curr_omega)
        mu_history_joint.append(curr_mu)

        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(
                f"  Epoch {ep+1:02d}/{epochs:02d} | Loss: {loss.item():.5e} | "
                f"beta: {curr_beta:.4f} (target: {true_beta:.2f}) | "
                f"omega^2: {curr_omega:.4f} (1.50) | mu: {curr_mu:.4f} (0.50)"
            )
    t_joint = time.perf_counter() - t0_joint

    # ---------------------------------------------------------
    # 2. Baseline Model: Fixed Misspecified Beta (\beta = 0.50)
    # ---------------------------------------------------------
    print("\n[Model 2] Training Fixed Misspecified Baseline (Frozen beta = 0.50)...")
    field_fixed = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
    model_fixed = NeuralFDE(
        vector_field=field_fixed,
        beta=init_beta,
        learnable_beta=False,
        default_tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
    )
    optimizer_fixed = torch.optim.Adam(model_fixed.parameters(), lr=0.045)
    scheduler_fixed = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fixed, T_max=epochs, eta_min=2e-3)

    loss_history_fixed = []
    t0_fixed = time.perf_counter()
    for ep in range(epochs):
        optimizer_fixed.zero_grad()
        pred_fixed = model_fixed(z0, t_grid, method="adamem", tol=1e-3)
        loss = torch.mean((pred_fixed - z_true) ** 2)
        loss.backward()
        optimizer_fixed.step()
        scheduler_fixed.step()
        loss_history_fixed.append(loss.item())

        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(
                f"  Epoch {ep+1:02d}/{epochs:02d} | Loss: {loss.item():.5e} | "
                f"beta: {init_beta:.4f} [FROZEN] | "
                f"omega^2: {field_fixed.omega2.item():.4f} | mu: {field_fixed.mu.item():.4f}"
            )
    t_fixed = time.perf_counter() - t0_fixed

    # ---------------------------------------------------------
    # 3. Reference Oracle Model: Known Ground-Truth Beta (\beta = 0.75)
    # ---------------------------------------------------------
    print("\n[Model 3] Training Known-Order Oracle Reference (Known beta* = 0.75)...")
    field_oracle = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
    model_oracle = NeuralFDE(
        vector_field=field_oracle,
        beta=true_beta,
        learnable_beta=False,
        default_tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
    )
    optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.035)
    scheduler_oracle = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_oracle, T_max=epochs, eta_min=1e-3)

    loss_history_oracle = []
    t0_oracle = time.perf_counter()
    for ep in range(epochs):
        optimizer_oracle.zero_grad()
        pred_oracle = model_oracle(z0, t_grid, method="adamem", tol=1e-3)
        loss = torch.mean((pred_oracle - z_true) ** 2)
        loss.backward()
        optimizer_oracle.step()
        scheduler_oracle.step()
        loss_history_oracle.append(loss.item())
    t_oracle = time.perf_counter() - t0_oracle

    # Evaluate final rollouts and dynamic memory modes
    from core.solvers.adaptive_soe import AdaptiveSOEFDESolver
    solver_eval = AdaptiveSOEFDESolver(
        beta=model_joint.get_beta_value(), tol=1e-3, K_init=8, K_min=4, K_max=24
    )
    sol_joint = solver_eval.solve(field_joint, z0, t_grid)
    final_pred_joint = sol_joint.z.detach()
    active_modes_joint = sol_joint.modes_history

    final_pred_fixed = model_fixed(z0, t_grid, method="adamem", tol=1e-3).detach()
    final_pred_oracle = model_oracle(z0, t_grid, method="adamem", tol=1e-3).detach()

    # Quantitative Summary Table
    final_beta = model_joint.get_beta_value()
    beta_err = abs(final_beta - true_beta)
    rel_err_pct = (beta_err / true_beta) * 100.0
    print("\n" + "=" * 82)
    print(f"{'Method / Configuration':<32} | {'Final Loss':<12} | {'Recovered beta':<16} | {'Status':<14}")
    print("-" * 82)
    print(f"{'Proposed Joint AdaMem-FDE':<32} | {loss_history_joint[-1]:<12.4e} | {final_beta:.4f} ({rel_err_pct:.2f}% err) | Converged")
    print(f"{'Fixed Misspecified (beta=0.50)':<32} | {loss_history_fixed[-1]:<12.4e} | 0.5000 (Frozen)   | Misspecified")
    print(f"{'Known-Order Oracle (beta=0.75)':<32} | {loss_history_oracle[-1]:<12.4e} | 0.7500 (Oracle)   | Reference")
    print("=" * 82)

    # ---------------------------------------------------------
    # Visualizations: 4-Panel Publication-Grade Figure
    # ---------------------------------------------------------
    t_np = t_grid.cpu().numpy()
    z_true_np = z_true.cpu().numpy()
    pred_joint_np = final_pred_joint.cpu().numpy()
    pred_fixed_np = final_pred_fixed.cpu().numpy()
    pred_oracle_np = final_pred_oracle.cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Trajectory Rollout Comparison
    axs[0, 0].plot(t_np, z_true_np[:, 0], "k-", lw=2.4, label=r"Ground Truth ($\beta^*=0.75$)")
    axs[0, 0].plot(
        t_np, pred_joint_np[:, 0], "b--", lw=2.0, label=rf"Joint AdaMem ($\beta={final_beta:.3f}$)"
    )
    axs[0, 0].plot(
        t_np, pred_fixed_np[:, 0], "r:", lw=1.8, label=r"Fixed Misspecified ($\beta=0.50$)"
    )
    axs[0, 0].set_xlabel("Time $t$", fontsize=11)
    axs[0, 0].set_ylabel(r"Displacement $x_1(t)$", fontsize=11)
    axs[0, 0].set_title(r"(a) Trajectory Reconstruction & Nonlocal Damping", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=9, loc="upper right")

    # Panel B: Fractional Order Discovery Trajectory
    epochs_arr = np.arange(1, epochs + 1)
    axs[0, 1].axhline(true_beta, color="k", linestyle="--", lw=2.0, label=r"True Order $\beta^* = 0.75$")
    axs[0, 1].plot(epochs_arr, beta_history_joint, "b-o", lw=2.0, ms=4, label=r"Discovered $\beta(t)$")
    # Exact 2% target bound: [0.75 * 0.98, 0.75 * 1.02] = [0.735, 0.765]
    axs[0, 1].fill_between(
        epochs_arr,
        true_beta * 0.98,
        true_beta * 1.02,
        color="green",
        alpha=0.15,
        label=r"$\pm 2\%$ Target Margin ($[0.735, 0.765]$)",
    )
    axs[0, 1].set_xlabel("Training Epoch", fontsize=11)
    axs[0, 1].set_ylabel(r"Fractional Order $\beta$", fontsize=11)
    axs[0, 1].set_title(r"(b) Analytical Adjoint Discovery of Fractional Order", fontsize=12, fontweight="bold")
    axs[0, 1].set_ylim(0.45, 0.82)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=9, loc="lower right")

    # Panel C: Loss Convergence Comparison
    axs[1, 0].semilogy(epochs_arr, loss_history_joint, "b-o", lw=1.8, ms=4, label=r"Proposed Joint AdaMem-FDE")
    axs[1, 0].semilogy(epochs_arr, loss_history_fixed, "r--s", lw=1.8, ms=4, label=r"Fixed Misspecified ($\beta=0.50$)")
    axs[1, 0].semilogy(epochs_arr, loss_history_oracle, "g-.^", lw=1.5, ms=4, label=r"Known Oracle ($\beta^*=0.75$)")
    axs[1, 0].set_xlabel("Training Epoch", fontsize=11)
    axs[1, 0].set_ylabel(r"Mean Squared Error $\mathcal{L}_{\mathrm{MSE}}$ (Log Scale)", fontsize=11)
    axs[1, 0].set_title(r"(c) Loss Convergence: Joint vs Misspecified Orders", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, alpha=0.3, which="both")
    axs[1, 0].legend(fontsize=9)

    # Panel D: Dynamic Memory Mode Allocation K(t)
    t_modes = np.linspace(0, 3.0, len(active_modes_joint))
    axs[1, 1].step(t_modes, active_modes_joint, where="post", color="purple", lw=2.0, label=r"AdaMem Active Modes $K(t)$")
    axs[1, 1].axhline(24, color="gray", linestyle=":", lw=1.5, label=r"Allocation Bound ($K_{\max}=24$)")
    axs[1, 1].set_xlabel("Time $t$", fontsize=11)
    axs[1, 1].set_ylabel(r"Active Exponential Modes $K(t)$", fontsize=11)
    axs[1, 1].set_title(r"(d) Dynamic Memory Mode Allocation $K(t)$", fontsize=12, fontweight="bold")
    axs[1, 1].set_ylim(0, 28)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_plot = os.path.join(save_dir, "phase8_joint_beta_discovery.png")
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"\n[Artifact Generated] Phase VIII publication plot saved to {out_plot}\n")


if __name__ == "__main__":
    run_phase8_experiment()
