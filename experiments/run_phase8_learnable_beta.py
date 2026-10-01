"""
Phase VIII: joint discovery of the fractional order beta and field parameters.

Protocol
--------
* Truth: damped fractional oscillator (beta*=0.75, omega^2=1.5, mu=0.5), generated on a
  10x finer grid with a tighter tolerance / larger K_max than the fitted model uses
  (avoids the "inverse crime" of fitting data produced by the identical solver).
* Per seed: independent observation noise AND independent random (omega^2, mu) init.
* Arms (all share data, init and field optimizer per seed):
    joint      beta learnable, beta0 in {0.30, 0.50, 0.90}
    fixed      beta frozen at 0.50 (misspecified)
    oracle     beta frozen at 0.75
    beta_only  (--diagnostic) field frozen at truth, only beta trained; isolates
               beta-theta coupling from optimizer issues
* Optimizers: field = Adam + cosine; beta = Adam(low momentum) + ReduceLROnPlateau.
* Runs execute in parallel, one single-threaded process each (float64, CPU).

Usage
-----
    python experiments/run_phase8_learnable_beta.py                  # 5 seeds, 1% noise
    python experiments/run_phase8_learnable_beta.py --noise 0.0
    python experiments/run_phase8_learnable_beta.py --noise 0.05 --diagnostic
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.adjoint.adamem_adjoint import adamem_integrate
from models.neural_fde import NeuralFDE

TRUE_BETA, TRUE_OMEGA2, TRUE_MU = 0.75, 1.50, 0.50
INIT_BETAS = (0.30, 0.50, 0.90)
T_END, N_STEPS, REFINE = 3.0, 60, 10

# Model/solver settings used when FITTING (truth uses tighter ones in make_truth)
FIT_TOL, FIT_K_INIT, FIT_K_MIN, FIT_K_MAX = 1e-3, 8, 4, 24

LR_FIELD, LR_BETA = 0.04, 0.05
BETA_ADAM_BETAS = (0.7, 0.99)


class ParameterizedOscillator(nn.Module):
    """D^b x1 = x2 ; D^b x2 = -omega2 * x1 - mu * x2."""

    def __init__(self, omega2: float, mu: float):
        super().__init__()
        self.omega2 = nn.Parameter(torch.tensor(omega2, dtype=torch.float64))
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float64))

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        x1, x2 = z[..., 0], z[..., 1]
        return torch.stack([x2, -self.omega2 * x1 - self.mu * x2], dim=-1)


# --------------------------------------------------------------------------- data
def make_truth(stride: int = REFINE):
    """High-fidelity reference trajectory, subsampled to the fitting grid.

    stride must divide N_STEPS * REFINE (=600): 10 -> 60 fit steps, 5 -> 120, 2 -> 300, 1 -> 600.
    """
    assert (N_STEPS * REFINE) % stride == 0, "fit stride must divide 600"
    fine = torch.linspace(0.0, T_END, N_STEPS * REFINE + 1, dtype=torch.float64)
    z0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    system = ParameterizedOscillator(TRUE_OMEGA2, TRUE_MU)
    with torch.no_grad():
        z_fine = adamem_integrate(
            system, z0, fine, beta=TRUE_BETA, tol=1e-5, K_init=16, K_min=4, K_max=40
        )
    return fine[::stride].clone(), z_fine[::stride].clone()


def make_observations(z_true: torch.Tensor, seed: int, sigma_rel: float) -> torch.Tensor:
    """Additive Gaussian noise, sigma = sigma_rel * per-state std. z(0) stays exact."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(z_true.shape, generator=gen, dtype=torch.float64)
    obs = z_true + sigma_rel * z_true.std(dim=0, keepdim=True) * noise
    obs[0] = z_true[0]
    return obs


def random_init(seed: int):
    gen = torch.Generator().manual_seed(seed + 10_000)
    u = torch.rand(2, generator=gen, dtype=torch.float64)
    return float(0.4 + 0.8 * u[0]), float(0.05 + 0.35 * u[1])  # omega2, mu


# ---------------------------------------------------------------------------- run
def _init_worker():
    torch.set_num_threads(1)


def run_task(spec: dict) -> dict:
    mode, seed, beta0, epochs = spec["mode"], spec["seed"], spec["beta0"], spec["epochs"]
    tol, k_max = spec["tol"], spec["k_max"]
    t_grid = torch.from_numpy(spec["t_grid"])
    z_obs = torch.from_numpy(spec["z_obs"])
    z0 = z_obs[0].clone()

    torch.manual_seed(seed)
    if mode == "beta_only":
        field = ParameterizedOscillator(TRUE_OMEGA2, TRUE_MU)
        field.requires_grad_(False)
    else:
        field = ParameterizedOscillator(*spec["init"])

    learn_beta = mode in ("joint", "beta_only")
    model = NeuralFDE(
        vector_field=field,
        beta=beta0,
        learnable_beta=learn_beta,
        default_tol=tol,
        K_init=min(FIT_K_INIT, k_max),
        K_min=FIT_K_MIN,
        K_max=k_max,
    )

    field_params = [p for p in field.parameters() if p.requires_grad]
    opt_f = sched_f = opt_b = sched_b = None
    if field_params:
        opt_f = torch.optim.Adam(field_params, lr=LR_FIELD)
        sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=epochs, eta_min=1e-3)
    if learn_beta:
        opt_b = torch.optim.Adam([model._raw_beta], lr=LR_BETA, betas=BETA_ADAM_BETAS)
        sched_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_b, factor=0.5, patience=10, min_lr=5e-3
        )

    loss_hist = np.full(epochs, np.nan)
    beta_hist = np.full(epochs, np.nan)
    diverged = False
    for ep in range(epochs):
        for o in (opt_f, opt_b):
            if o is not None:
                o.zero_grad()
        pred = model(z0, t_grid, method="adamem", tol=tol)
        loss = torch.mean((pred - z_obs) ** 2)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        for o in (opt_f, opt_b):
            if o is not None:
                o.step()
        if sched_f is not None:
            sched_f.step()
        if sched_b is not None:
            sched_b.step(loss.item())
        loss_hist[ep] = loss.item()
        beta_hist[ep] = model.get_beta_value()

    return {
        "mode": mode,
        "seed": seed,
        "beta0": beta0,
        "loss_hist": loss_hist.tolist(),
        "beta_hist": beta_hist.tolist(),
        "final_beta": float(model.get_beta_value()),
        "omega2": float(field.omega2.item()),
        "mu": float(field.mu.item()),
        "diverged": diverged,
    }


# ------------------------------------------------------------------- aggregation
def group_key(r: dict) -> str:
    if r["mode"] == "joint":
        return f"joint_b0={r['beta0']:.2f}"
    if r["mode"] == "beta_only":
        return f"beta_only_b0={r['beta0']:.2f}"
    return r["mode"]


def _last_finite(hist) -> float:
    a = np.asarray(hist, dtype=float)
    a = a[np.isfinite(a)]
    return float(a[-1]) if a.size else float("nan")


def summarize(results, sigma):
    groups = {}
    for r in results:
        groups.setdefault(group_key(r), []).append(r)

    print("\n" + "=" * 112)
    print(f"  Phase VIII summary | noise sigma = {sigma:g} x state std | n = runs per row")
    print("=" * 112)
    print(f"{'arm':<22}{'n':>3}  {'final beta (mean±std)':<24}{'|err| %':>8}  "
          f"{'in ±2%':>7}  {'final loss median [min, max]':<38}{'div':>4}")
    print("-" * 112)
    for key, rs in groups.items():
        b = np.array([r["final_beta"] for r in rs])
        l = np.array([_last_finite(r["loss_hist"]) for r in rs])
        err = np.abs(b - TRUE_BETA) / TRUE_BETA * 100
        hit = int(np.sum(err <= 2.0))
        div = sum(r["diverged"] for r in rs)
        beta_str = f"{b.mean():.4f} ± {b.std():.4f}"
        loss_str = f"{np.nanmedian(l):.2e} [{np.nanmin(l):.2e}, {np.nanmax(l):.2e}]"
        print(f"{key:<22}{len(rs):>3}  {beta_str:<24}{err.mean():>8.2f}  "
              f"{hit:>3}/{len(rs):<3}  {loss_str:<38}{div:>4}")
    print("=" * 112)
    return groups


def stack(rs, field):
    return np.array([r[field] for r in rs], dtype=float)


def plot(groups, sigma, epochs, out_path):
    colors = {0.30: "tab:purple", 0.50: "tab:blue", 0.90: "tab:green"}
    ep = np.arange(1, epochs + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    # (a) beta(t): mean ± std over seeds
    ax = axs[0]
    ax.axhline(TRUE_BETA, color="k", ls="--", lw=1.8, label=r"$\beta^*=0.75$")
    ax.axhspan(TRUE_BETA * 0.98, TRUE_BETA * 1.02, color="green", alpha=0.15, label=r"$\pm2\%$")
    for b0 in INIT_BETAS:
        rs = groups.get(f"joint_b0={b0:.2f}")
        if not rs:
            continue
        B = stack(rs, "beta_hist")
        m, s = np.nanmean(B, 0), np.nanstd(B, 0)
        ax.plot(ep, m, color=colors[b0], lw=2, label=rf"$\beta_0={b0:.2f}$ (n={len(rs)})")
        ax.fill_between(ep, m - s, m + s, color=colors[b0], alpha=0.18)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$\beta$")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(r"(a) $\beta$ trajectory, mean $\pm$ std over seeds", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) final beta per run
    ax = axs[1]
    ax.axhline(TRUE_BETA, color="k", ls="--", lw=1.8)
    ax.axhspan(TRUE_BETA * 0.98, TRUE_BETA * 1.02, color="green", alpha=0.15)
    keys = [k for k in groups if k.startswith("joint") or k.startswith("beta_only")]
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        vals = np.array([r["final_beta"] for r in groups[k]])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=36, alpha=0.8)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="k", lw=2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_b0=", "\n$\\beta_0$=") for k in keys], fontsize=8)
    ax.set_ylabel("Final recovered $\\beta$")
    ax.set_title("(b) Final $\\beta$ per run (bar = mean)", fontweight="bold")
    ax.grid(alpha=0.3)

    # (c) loss: median with min-max band
    ax = axs[2]
    arms = [(f"joint_b0={b0:.2f}", colors[b0], rf"Joint $\beta_0={b0:.2f}$") for b0 in INIT_BETAS]
    arms += [("fixed", "tab:red", r"Fixed $\beta=0.50$"), ("oracle", "k", r"Oracle $\beta^*=0.75$")]
    for key, c, label in arms:
        rs = groups.get(key)
        if not rs:
            continue
        L = stack(rs, "loss_hist")
        ax.semilogy(ep, np.nanmedian(L, 0), color=c, lw=1.8, label=label)
        ax.fill_between(ep, np.nanmin(L, 0), np.nanmax(L, 0), color=c, alpha=0.12)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("MSE vs observations")
    ax.set_title("(c) Loss, median with min-max band", fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(f"Phase VIII: learnable $\\beta$ (noise $\\sigma$={sigma:g}, "
                 f"{len(groups.get('oracle', []))} seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.01, help="relative noise std (0 = noiseless)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--diagnostic", action="store_true", help="add beta-only (field frozen at truth) arm")
    ap.add_argument("--fit-stride", type=int, default=REFINE,
                    help="fit grid = truth grid subsampled by this (10->60 steps, 5->120, 2->300, 1->600)")
    ap.add_argument("--fit-tol", type=float, default=FIT_TOL, help="controller tol of the fitted model")
    ap.add_argument("--fit-kmax", type=int, default=FIT_K_MAX, help="K_max of the fitted model")
    ap.add_argument("--save-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    seeds = [42, 101, 202, 303, 404, 505, 606, 707][: args.seeds]
    t_grid, z_true = make_truth(args.fit_stride)
    t_np = t_grid.numpy()

    specs = []
    for s in seeds:
        z_obs = make_observations(z_true, s, args.noise).numpy()
        init = random_init(s)
        base = {"seed": s, "epochs": args.epochs, "t_grid": t_np, "z_obs": z_obs, "init": init,
                "tol": args.fit_tol, "k_max": args.fit_kmax}
        for b0 in INIT_BETAS:
            specs.append({**base, "mode": "joint", "beta0": b0})
        specs.append({**base, "mode": "fixed", "beta0": 0.50})
        specs.append({**base, "mode": "oracle", "beta0": TRUE_BETA})
        if args.diagnostic:
            for b0 in INIT_BETAS:
                specs.append({**base, "mode": "beta_only", "beta0": b0})

    n_fit = len(t_np) - 1
    print(f"Phase VIII | {len(specs)} runs | {args.workers} workers | {args.epochs} epochs | "
          f"noise={args.noise:g} | seeds={seeds} | fit: N={n_fit}, tol={args.fit_tol:g}, K_max={args.fit_kmax}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = list(pool.map(run_task, specs))
    print(f"Done in {time.time() - t0:.1f}s")

    groups = summarize(results, args.noise)

    tag = f"noise{args.noise:g}_N{n_fit}_tol{args.fit_tol:g}_K{args.fit_kmax}"
    with open(os.path.join(args.save_dir, f"phase8_multiseed_{tag}.json"), "w") as f:
        json.dump(results, f)
    out_png = os.path.join(args.save_dir, f"phase8_joint_beta_discovery_{tag}.png")
    plot(groups, args.noise, args.epochs, out_png)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()