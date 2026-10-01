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


def train_single_beta_run(init_beta: float, seed: int, epochs: int, t_grid, z0, z_true):
    torch.manual_seed(seed)
    field = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
    model = NeuralFDE(
        vector_field=field,
        beta=init_beta,
        learnable_beta=True,
        default_tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=24,
    )
    opt_field = torch.optim.Adam(field.parameters(), lr=0.04)
    sched_field = torch.optim.lr_scheduler.CosineAnnealingLR(opt_field, T_max=epochs, eta_min=1e-3)
    opt_beta = torch.optim.Adam([model._raw_beta], lr=0.18)
    sched_beta = torch.optim.lr_scheduler.StepLR(opt_beta, step_size=40, gamma=0.7)

    loss_hist = []
    beta_hist = []

    for ep in range(epochs):
        opt_field.zero_grad()
        opt_beta.zero_grad()
        pred = model(z0, t_grid, method="adamem", tol=1e-3)
        loss = torch.mean((pred - z_true) ** 2)
        loss.backward()
        opt_field.step()
        opt_beta.step()
        sched_field.step()
        sched_beta.step()

        loss_hist.append(loss.item())
        beta_hist.append(model.get_beta_value())

    final_pred = model(z0, t_grid, method="adamem", tol=1e-3).detach()
    return loss_hist, beta_hist, final_pred, model.get_beta_value(), field.omega2.item(), field.mu.item()


def run_phase8_experiment(save_dir: str = "results", epochs: int = 70):
    os.makedirs(save_dir, exist_ok=True)
    true_beta = 0.75
    init_betas = [0.30, 0.50, 0.90]
    seeds = [42, 101, 202]

    print("\n" + "=" * 84)
    print("  PHASE VIII: LEARNABLE FRACTIONAL ORDER (BETA) MULTI-INITIALIZATION BENCHMARK")
    print(f"  Target System: Damped Fractional Oscillator (True beta* = {true_beta:.2f})")
    print(f"  Initial Orders: beta_0 in {init_betas}, Seeds: {seeds}, Epochs: {epochs}")
    print("=" * 84)

    t_grid, z0, z_true = generate_ground_truth(true_beta=true_beta, T=3.0, num_steps=60)

    # 1. Multi-beta0, multi-seed training
    results_by_beta0 = {}
    for b0 in init_betas:
        print(f"\n--- Testing Initial Order beta_0 = {b0:.2f} (Delta beta = {b0 - true_beta:+.2f}) ---")
        runs = []
        for s in seeds:
            lh, bh, fp, final_b, final_w, final_m = train_single_beta_run(b0, s, epochs, t_grid, z0, z_true)
            runs.append({
                "seed": s, "loss_hist": lh, "beta_hist": bh, "final_pred": fp,
                "final_beta": final_b, "final_omega": final_w, "final_mu": final_m,
                "rel_err": abs(final_b - true_beta) / true_beta * 100.0,
            })
            print(f"  Seed {s:3d} | Final Loss: {lh[-1]:.4e} | Recovered beta: {final_b:.4f} (err: {runs[-1]['rel_err']:.2f}%) | omega^2: {final_w:.4f} | mu: {final_m:.4f}")
        results_by_beta0[b0] = runs

    # 2. Fixed Misspecified Baseline (beta = 0.50 frozen, 3 seeds)
    print("\n--- Training Fixed Misspecified Baseline (Frozen beta = 0.50) ---")
    fixed_losses = []
    fixed_preds = []
    for s in seeds:
        torch.manual_seed(s)
        field_fixed = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
        model_fixed = NeuralFDE(vector_field=field_fixed, beta=0.50, learnable_beta=False, default_tol=1e-3, K_init=8, K_min=4, K_max=24)
        opt_fixed = torch.optim.Adam(model_fixed.parameters(), lr=0.04)
        sched_fixed = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fixed, T_max=epochs, eta_min=1e-3)
        lh = []
        for ep in range(epochs):
            opt_fixed.zero_grad()
            pred = model_fixed(z0, t_grid, method="adamem", tol=1e-3)
            loss = torch.mean((pred - z_true) ** 2)
            loss.backward()
            opt_fixed.step()
            sched_fixed.step()
            lh.append(loss.item())
        fixed_losses.append(lh)
        fixed_preds.append(model_fixed(z0, t_grid, method="adamem", tol=1e-3).detach())
        print(f"  Seed {s:3d} | Final Loss: {lh[-1]:.4e} | beta: 0.5000 [FROZEN]")

    # 3. Known-Order Oracle Reference (beta* = 0.75 known, 3 seeds)
    print("\n--- Training Known-Order Oracle Reference (Known beta* = 0.75) ---")
    oracle_losses = []
    oracle_preds = []
    for s in seeds:
        torch.manual_seed(s)
        field_oracle = ParameterizedOscillator(init_omega2=0.80, init_mu=0.20)
        model_oracle = NeuralFDE(vector_field=field_oracle, beta=true_beta, learnable_beta=False, default_tol=1e-3, K_init=8, K_min=4, K_max=24)
        opt_oracle = torch.optim.Adam(model_oracle.parameters(), lr=0.035)
        sched_oracle = torch.optim.lr_scheduler.CosineAnnealingLR(opt_oracle, T_max=epochs, eta_min=1e-3)
        lh = []
        for ep in range(epochs):
            opt_oracle.zero_grad()
            pred = model_oracle(z0, t_grid, method="adamem", tol=1e-3)
            loss = torch.mean((pred - z_true) ** 2)
            loss.backward()
            opt_oracle.step()
            sched_oracle.step()
            lh.append(loss.item())
        oracle_losses.append(lh)
        oracle_preds.append(model_oracle(z0, t_grid, method="adamem", tol=1e-3).detach())
        print(f"  Seed {s:3d} | Final Loss: {lh[-1]:.4e} | beta: 0.7500 [ORACLE]")

    # Summary Statistics Table
    print("\n" + "=" * 84)
    print(f"{'Initialization / Baseline':<32} | {'Final Loss (Mean +/- Std)':<26} | {'Recovered beta':<20}")
    print("-" * 84)
    for b0 in init_betas:
        b_vals = [r["final_beta"] for r in results_by_beta0[b0]]
        l_vals = [r["loss_hist"][-1] for r in results_by_beta0[b0]]
        err_vals = [r["rel_err"] for r in results_by_beta0[b0]]
        label = f"Joint AdaMem (beta0={b0:.2f})"
        print(f"{label:<32} | {np.mean(l_vals):.4e} +/- {np.std(l_vals):.4e}    | {np.mean(b_vals):.4f} +/- {np.std(b_vals):.4f} ({np.mean(err_vals):.1f}%)")

    l_fix = [lh[-1] for lh in fixed_losses]
    l_orc = [lh[-1] for lh in oracle_losses]
    print(f"{'Fixed Misspecified (beta=0.50)':<32} | {np.mean(l_fix):.4e} +/- {np.std(l_fix):.4e}    | 0.5000 [Frozen] (33.3%)")
    print(f"{'Known-Order Oracle (beta=0.75)':<32} | {np.mean(l_orc):.4e} +/- {np.std(l_orc):.4e}    | 0.7500 [Oracle] (0.0%)")
    print("=" * 84)

    # ---------------------------------------------------------
    # Visualizations: 4-Panel Publication-Grade Figure
    # ---------------------------------------------------------
    t_np = t_grid.cpu().numpy()
    z_true_np = z_true.cpu().numpy()
    epochs_arr = np.arange(1, epochs + 1)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Trajectory Rollout Comparison
    axs[0, 0].plot(t_np, z_true_np[:, 0], "k-", lw=2.4, label=r"Ground Truth ($\beta^*=0.75$)")
    colors_b = {0.30: "purple", 0.50: "blue", 0.90: "teal"}
    for b0 in init_betas:
        pred_mean = np.mean([r["final_pred"][:, 0].cpu().numpy() for r in results_by_beta0[b0]], axis=0)
        b_mean = np.mean([r["final_beta"] for r in results_by_beta0[b0]])
        axs[0, 0].plot(t_np, pred_mean, color=colors_b[b0], linestyle="--", lw=1.8, label=rf"Recovered ($\beta_0={b0:.2f} \to {b_mean:.3f}$)")
    axs[0, 0].plot(t_np, np.mean([p[:, 0].cpu().numpy() for p in fixed_preds], axis=0), "r:", lw=1.8, label=r"Fixed Misspecified ($\beta=0.50$)")
    axs[0, 0].set_xlabel("Time $t$", fontsize=11)
    axs[0, 0].set_ylabel(r"Displacement $x_1(t)$", fontsize=11)
    axs[0, 0].set_title(r"(a) Trajectory Reconstruction Across Initializations", fontsize=12, fontweight="bold")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8, loc="upper right")

    # Panel B: Fractional Order Discovery Trajectory Fan
    axs[0, 1].axhline(true_beta, color="k", linestyle="--", lw=2.0, label=r"True Order $\beta^* = 0.75$")
    axs[0, 1].fill_between(
        epochs_arr,
        true_beta * 0.98,
        true_beta * 1.02,
        color="green",
        alpha=0.15,
        label=r"$\pm 2\%$ Target Margin ($[0.735, 0.765]$)",
    )
    for b0 in init_betas:
        b_arrs = np.array([r["beta_hist"] for r in results_by_beta0[b0]])
        b_mean = np.mean(b_arrs, axis=0)
        b_std = np.std(b_arrs, axis=0)
        axs[0, 1].plot(epochs_arr, b_mean, color=colors_b[b0], lw=2.0, label=rf"Init $\beta_0={b0:.2f} \to {b_mean[-1]:.3f}\pm{b_std[-1]:.3f}$")
        axs[0, 1].fill_between(epochs_arr, b_mean - b_std, b_mean + b_std, color=colors_b[b0], alpha=0.15)
    axs[0, 1].set_xlabel("Training Epoch", fontsize=11)
    axs[0, 1].set_ylabel(r"Fractional Order $\beta(t)$", fontsize=11)
    axs[0, 1].set_title(r"(b) Convergence of $\beta(t)$ from Multiple Initializations", fontsize=12, fontweight="bold")
    axs[0, 1].set_ylim(0.25, 0.95)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=8, loc="center right")

    # Panel C: Loss Convergence Comparison
    for b0 in init_betas:
        l_arrs = np.array([r["loss_hist"] for r in results_by_beta0[b0]])
        l_mean = np.mean(l_arrs, axis=0)
        axs[1, 0].semilogy(epochs_arr, l_mean, color=colors_b[b0], lw=1.8, label=rf"Joint AdaMem ($\beta_0={b0:.2f}$)")
    axs[1, 0].semilogy(epochs_arr, np.mean(fixed_losses, axis=0), "r--", lw=1.8, label=r"Fixed Misspecified ($\beta=0.50$)")
    axs[1, 0].semilogy(epochs_arr, np.mean(oracle_losses, axis=0), "g-.^", lw=1.5, ms=3, label=r"Known Oracle ($\beta^*=0.75$)")
    axs[1, 0].set_xlabel("Training Epoch", fontsize=11)
    axs[1, 0].set_ylabel(r"MSE Loss (Log Scale)", fontsize=11)
    axs[1, 0].set_title(r"(c) Multi-Seed Loss Convergence", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, alpha=0.3, which="both")
    axs[1, 0].legend(fontsize=8)

    # Panel D: Dynamic Memory Mode Allocation K(t)
    from core.solvers.adaptive_soe import AdaptiveSOEFDESolver
    solver_eval = AdaptiveSOEFDESolver(beta=0.75, tol=1e-3, K_init=8, K_min=4, K_max=24)
    sol_eval = solver_eval.solve(ParameterizedOscillator(1.50, 0.50), z0, t_grid)
    t_modes = np.linspace(0, 3.0, len(sol_eval.modes_history))
    axs[1, 1].step(t_modes, sol_eval.modes_history, where="post", color="purple", lw=2.0, label=r"Active Modes $K(t)$")
    axs[1, 1].axhline(24, color="gray", linestyle=":", lw=1.5, label=r"Max Modes ($K_{\max}=24$)")
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
