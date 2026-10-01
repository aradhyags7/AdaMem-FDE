"""
Phase IV & V Experiment: Neural Fractional Differential Equation Training & Ablation Study.

Compares training performance across multiple random seeds (N_seeds = 5):
1. AdaMem-FDE (Proposed: Adaptive Memory + Adjoint-Consistent R^T Transitions)
2. Baseline 4 (Ablation: Adaptive Memory without Adjoint R^T Transitions)

Monitors:
- Training loss convergence mean +/- 1 std error band
- Trajectory reconstruction fidelity against ground truth Duffing dynamics
- Adjoint gradient error distribution (E_g) vs two-sided finite difference reference
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.systems import FractionalDuffing
from core.adjoint.adamem_adjoint import adamem_integrate
from core.solvers.fixed_soe import FixedSOEFDESolver
from models.neural_fde import NeuralFDE


class StructuredDuffingField(torch.nn.Module):
    """
    Second-order neural vector field parameterizing:
        D^\beta x_1 = x_2
        D^\beta x_2 = MLP_\theta(x_1, x_2, t)
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(dtype=torch.float64)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=0.6)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        x1 = z[..., 0:1]
        x2 = z[..., 1:2]
        t_ten = torch.full_like(x1, fill_value=t)
        inp = torch.cat([x1, x2, t_ten], dim=-1)
        dx2 = self.net(inp)
        dx1 = x2
        return torch.cat([dx1, dx2], dim=-1)


def generate_ground_truth_data(beta=0.85, T=3.0, num_steps=60):
    t_grid = torch.linspace(0.0, T, num_steps + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    system = FractionalDuffing(beta=beta)
    solver = FixedSOEFDESolver(beta=beta, num_modes=32)
    z_true = solver.solve(system.rhs, z0, t_grid).detach()
    return t_grid, z0, z_true


def train_single_seed(seed: int, method: str, t_grid, z0, z_true, beta=0.85, epochs=35, lr=0.01):
    torch.manual_seed(seed)
    vf = StructuredDuffingField(hidden_dim=32)
    model = NeuralFDE(vector_field=vf, beta=beta, default_tol=1e-3, K_init=8, K_min=4, K_max=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for ep in range(epochs):
        optimizer.zero_grad()
        z_pred = model(z0, t_grid, method=method, tol=1e-3)
        loss = torch.mean((z_pred - z_true) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())

    final_pred = model(z0, t_grid, method=method, tol=1e-3).detach()
    return loss_history, final_pred


def evaluate_gradient_errors(t_grid, z0, z_true, seeds, beta=0.85):
    """Evaluates adjoint gradient relative error vs finite differences on the neural network across seeds."""
    h = 1e-6
    err_prop, err_naive = [], []

    for s in seeds:
        torch.manual_seed(s)
        vf = StructuredDuffingField(hidden_dim=32)
        p = list(vf.parameters())[0]

        with torch.no_grad():
            p.data[0, 0] += h
            z_plus = adamem_integrate(vf, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
            lp = 0.5 * torch.sum((z_plus - z_true) ** 2).item()
            p.data[0, 0] -= 2 * h
            z_minus = adamem_integrate(vf, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True)
            lm = 0.5 * torch.sum((z_minus - z_true) ** 2).item()
            p.data[0, 0] += h
        g_fd = (lp - lm) / (2.0 * h)

        # Proposed with R^T
        vf.zero_grad()
        zp = adamem_integrate(vf, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=True, parameters=tuple(vf.parameters()))
        (0.5 * torch.sum((zp - z_true) ** 2)).backward()
        gp = p.grad[0, 0].item()

        # Naive without R^T
        vf.zero_grad()
        zn = adamem_integrate(vf, z0, t_grid, beta=beta, tol=1e-3, K_init=8, K_min=4, K_max=24, use_adjoint_jump=False, parameters=tuple(vf.parameters()))
        (0.5 * torch.sum((zn - z_true) ** 2)).backward()
        gn = p.grad[0, 0].item()

        ep = abs(gp - g_fd) / abs(g_fd)
        en = abs(gn - g_fd) / abs(g_fd)
        err_prop.append(ep)
        err_naive.append(en)

    return np.array(err_prop), np.array(err_naive)


def run_phase4_experiment(save_dir: str = "results", epochs: int = 35):
    os.makedirs(save_dir, exist_ok=True)
    from experiments.archive_utils import archive_previous_results
    archive_previous_results("phase4_training", ["phase4_neural_fde_training.png", "phase4_neural_fde_training.json"], save_dir=save_dir)
    beta = 0.85
    seeds = [42, 101, 202, 303, 404]

    print("\n" + "=" * 78)
    print(f"  PHASE IV & V: NEURAL FDE TRAINING & MULTI-SEED ABLATION STUDY (beta={beta})")
    print(f"  Seeds: {seeds}, Epochs per seed: {epochs}")
    print("=" * 78)

    t_grid, z0, z_true = generate_ground_truth_data(beta=beta, T=3.0, num_steps=60)

    # 1. Multi-seed training for Proposed AdaMem-FDE (with R^T jump)
    print("\n--- Training Proposed AdaMem-FDE (with R^T Jump) Across 5 Seeds ---")
    prop_losses = []
    prop_preds = []
    t0 = time.perf_counter()
    for s in seeds:
        lh, fp = train_single_seed(s, "adamem", t_grid, z0, z_true, beta=beta, epochs=epochs)
        prop_losses.append(lh)
        prop_preds.append(fp)
        print(f"  Seed {s:3d} | Final Loss: {lh[-1]:.5e}")
    time_prop = time.perf_counter() - t0

    # 2. Multi-seed training for Baseline 4 (Ablation: without R^T jump)
    print("\n--- Training Baseline 4 (Naive without R^T Jump) Across 5 Seeds ---")
    naive_losses = []
    naive_preds = []
    t0 = time.perf_counter()
    for s in seeds:
        lh, fp = train_single_seed(s, "naive_adaptive", t_grid, z0, z_true, beta=beta, epochs=epochs)
        naive_losses.append(lh)
        naive_preds.append(fp)
        print(f"  Seed {s:3d} | Final Loss: {lh[-1]:.5e}")
    time_naive = time.perf_counter() - t0

    # Convert to numpy arrays: shape (num_seeds, epochs)
    arr_prop = np.array(prop_losses)
    arr_naive = np.array(naive_losses)

    mean_prop = np.mean(arr_prop, axis=0)
    std_prop = np.std(arr_prop, axis=0)
    mean_naive = np.mean(arr_naive, axis=0)
    std_naive = np.std(arr_naive, axis=0)

    # 3. Quantitative Gradient Error Evaluation
    print("\n--- Evaluating Adjoint Gradient Error Across Initializations ---")
    err_prop_arr, err_naive_arr = evaluate_gradient_errors(t_grid, z0, z_true, seeds=seeds, beta=beta)
    print(f"Proposed Relative Gradient Error: {np.mean(err_prop_arr)*100:.2f}% +/- {np.std(err_prop_arr)*100:.2f}%")
    print(f"Naive Relative Gradient Error:    {np.mean(err_naive_arr)*100:.2f}% +/- {np.std(err_naive_arr)*100:.2f}%")
    print(f"Gradient Error Reduction:         {np.mean(err_naive_arr) / np.mean(err_prop_arr):.2f}x lower error with R^T jump")

    print("\n" + "=" * 78)
    print(f"{'Method / Configuration':<30} | {'Final Loss (Mean +/- Std)':<26} | {'Grad Error':<14}")
    print("-" * 78)
    print(f"{'Proposed AdaMem-FDE (R^T)':<30} | {mean_prop[-1]:.4e} +/- {std_prop[-1]:.4e}    | {np.mean(err_prop_arr)*100:.1f}% +/- {np.std(err_prop_arr)*100:.1f}%")
    print(f"{'Baseline 4 (No Jump)':<30} | {mean_naive[-1]:.4e} +/- {std_naive[-1]:.4e}    | {np.mean(err_naive_arr)*100:.1f}% +/- {np.std(err_naive_arr)*100:.1f}%")
    print("=" * 78)

    # Visualization: 3-panel publication figure
    t_np = t_grid.cpu().numpy()
    z_true_np = z_true.cpu().numpy()
    epochs_arr = np.arange(1, epochs + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: Multi-Seed Loss Convergence with Confidence Bands
    ax1.semilogy(epochs_arr, mean_prop, "b-", lw=2.0, label=rf"Proposed ($R^T$ Jump): {mean_prop[-1]:.2e}$\pm${std_prop[-1]:.2e}")
    ax1.fill_between(epochs_arr, np.maximum(1e-5, mean_prop - std_prop), mean_prop + std_prop, color="blue", alpha=0.18)

    ax1.semilogy(epochs_arr, mean_naive, "r--", lw=1.8, label=rf"Baseline 4 (No Jump): {mean_naive[-1]:.2e}$\pm${std_naive[-1]:.2e}")
    ax1.fill_between(epochs_arr, np.maximum(1e-5, mean_naive - std_naive), mean_naive + std_naive, color="red", alpha=0.18)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel(r"MSE Loss (Mean $\pm$ Std, $N_{\mathrm{seeds}}=5$)", fontsize=10)
    ax1.set_title("(a) Multi-Seed Training Loss Convergence", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: Learned Trajectory Rollout vs Ground Truth
    best_pred_prop = prop_preds[0].cpu().numpy()
    best_pred_naive = naive_preds[0].cpu().numpy()
    ax2.plot(t_np, z_true_np[:, 0], "k-", lw=2.2, label=r"Ground Truth $x_1(t)$")
    ax2.plot(t_np, best_pred_prop[:, 0], "b--", lw=1.8, label=r"AdaMem Prediction ($R^T$)")
    ax2.plot(t_np, best_pred_naive[:, 0], "r:", lw=1.5, label=r"Baseline 4 Prediction")
    ax2.set_xlabel(r"Time $t$", fontsize=11)
    ax2.set_ylabel(r"State $x_1(t)$", fontsize=11)
    ax2.set_title("(b) Learned Physical Trajectory Tracking", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower left")

    # Panel 3: Relative Adjoint Gradient Error Distribution
    x_pos = np.arange(len(err_prop_arr))
    width = 0.35
    ax3.bar(x_pos - width / 2, err_prop_arr * 100, width, label=r"Proposed ($R^T$ Jump)", color="royalblue", alpha=0.85)
    ax3.bar(x_pos + width / 2, err_naive_arr * 100, width, label=r"Baseline 4 (No Jump)", color="salmon", alpha=0.85)
    ax3.axhline(np.mean(err_prop_arr) * 100, color="blue", linestyle="--", lw=1.5, label=rf"Proposed: {np.mean(err_prop_arr)*100:.1f}$\pm${np.std(err_prop_arr)*100:.1f}%")
    ax3.axhline(np.mean(err_naive_arr) * 100, color="red", linestyle=":", lw=1.5, label=rf"Baseline: {np.mean(err_naive_arr)*100:.1f}$\pm${np.std(err_naive_arr)*100:.1f}%")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=9)
    ax3.set_ylabel(r"Relative Gradient Error $E_g$ (%)", fontsize=10)
    ax3.set_title(r"(c) Adjoint Gradient Error ($E_g$ on NN)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "phase4_neural_fde_training.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Save structured benchmark data for research paper reporting
    import json
    data_record = {
        "seeds": seeds,
        "epochs": epochs,
        "proposed": {
            "loss_histories": [l for l in prop_losses],
            "final_loss_mean": float(np.mean([l[-1] for l in prop_losses])),
            "final_loss_std": float(np.std([l[-1] for l in prop_losses])),
            "gradient_error_percent": (err_prop_arr * 100).tolist(),
            "gradient_error_mean": float(np.mean(err_prop_arr) * 100),
            "gradient_error_std": float(np.std(err_prop_arr) * 100),
        },
        "baseline_no_jump": {
            "loss_histories": [l for l in naive_losses],
            "final_loss_mean": float(np.mean([l[-1] for l in naive_losses])),
            "final_loss_std": float(np.std([l[-1] for l in naive_losses])),
            "gradient_error_percent": (err_naive_arr * 100).tolist(),
            "gradient_error_mean": float(np.mean(err_naive_arr) * 100),
            "gradient_error_std": float(np.std(err_naive_arr) * 100),
        },
        "gradient_error_reduction_factor": float(np.mean(err_naive_arr) / np.mean(err_prop_arr)),
    }
    json_path = os.path.join(save_dir, "phase4_neural_fde_training.json")
    with open(json_path, "w") as f:
        json.dump(data_record, f, indent=2)

    print(f"\n[Artifact Saved] Phase IV multi-seed training figure saved to {plot_path}")
    print(f"[Artifact Saved] Phase IV multi-seed training data saved to {json_path}\n")


if __name__ == "__main__":
    run_phase4_experiment()
