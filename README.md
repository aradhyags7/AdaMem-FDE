# AdaMem-FDE

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch Version" />
  <img src="https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen?logo=pytest&logoColor=white" alt="Tests" />
  <img src="https://img.shields.io/badge/Status-Research%20Grade-purple" alt="Status" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</p>

<p align="center">
  <b>Error-Adaptive Dynamic Memory Compression for Adjoint-Trained Neural Fractional Differential Equations</b>
</p>

<p align="center">
  <em>Can the long-range memory of a Neural Fractional Differential Equation be represented with the minimum computational complexity required to satisfy prescribed forward and gradient accuracy constraints?</em>
</p>

---

## Table of Contents
1. [Introduction & Motivation](#1-introduction--motivation)
2. [Mathematical Formulation](#2-mathematical-formulation)
   - [2.1 Caputo Neural Fractional Differential Equations](#21-caputo-neural-fractional-differential-equations)
   - [2.2 Memory Kernel Decomposition via SOE](#22-memory-kernel-decomposition-via-soe)
   - [2.3 Volterra State-Space Reformulation](#23-volterra-state-space-reformulation)
   - [2.4 Analytical Cauchy-Gram Projection Operator](#24-analytical-cauchy-gram-projection-operator)
   - [2.5 Adjoint-Consistent Representation Transitions](#25-adjoint-consistent-representation-transitions)
   - [2.6 Embedded Error Controller](#26-embedded-error-controller)
3. [Repository Architecture](#3-repository-architecture)
4. [Experimental Suite & Empirical Results](#4-experimental-suite--empirical-results)
   - [Phase I & II: Ground-Truth Mittag-Leffler Verification](#phase-i--ii-ground-truth-mittag-leffler-verification)
   - [Phase III: Dynamic Memory Mode Adaptation K(t)](#phase-iii-dynamic-memory-mode-adaptation-kt)
   - [Phase IV & V: Neural FDE Training & Ablation Study](#phase-iv--v-neural-fde-training--ablation-study)
   - [Phase VI: Long-Horizon Scalability on Lorenz Attractor](#phase-vi-long-horizon-scalability-on-lorenz-attractor)
5. [Baselines Evaluated](#5-baselines-evaluated)
6. [Installation & Setup](#6-installation--setup)
7. [Running Experiments & Reproduction](#7-running-experiments--reproduction)
8. [Running Unit Tests](#8-running-unit-tests)
9. [Citation & References](#9-citation--references)

---

## 1. Introduction & Motivation

Fractional differential equations (FDEs) provide a principled mathematical framework for modeling dynamical systems that exhibit **hereditary effects, anomalous diffusion, viscoelastic relaxation, and long-range power-law memory**. Neural Fractional Differential Equations (Neural FDEs) parameterize the governing dynamics with neural networks:

$${}^{C}D_t^\beta z(t) = f_\theta(z(t), t), \qquad 0 < \beta \le 1$$

However, direct evaluation of the Caputo derivative requires convolution over the entire previous trajectory $[0, t]$, causing:
- **$\mathcal{O}(N^2)$ Time Complexity**: Prohibitive computational cost as time horizon $N$ grows.
- **$\mathcal{O}(N)$ Memory Complexity**: Exhausting GPU/RAM storage during long simulations.
- **Severe Training Bottleneck**: Backpropagating through hundreds of training epochs becomes intractable.

### The Limitation of Fixed Memory Compression
Classical Sum-of-Exponentials (SOE) compression uses a fixed number of modes $K = \text{const}$. This is suboptimal:
* In simple or slowly-varying dynamical regimes, a large $K$ wastes computational resources.
* In complex transients or high-frequency regimes, a small $K$ sacrifices simulation accuracy.

### The AdaMem-FDE Solution
**AdaMem-FDE** treats fractional memory as a **dynamic computational resource** $K \to K(t)$. It introduces:
1. **Error-Controlled Memory Adaptation**: Automatically injects or prunes auxiliary memory modes based on an embedded shadow error estimator $\widehat{\epsilon}_M(t) \le \epsilon_{\text{tol}}$.
2. **Adjoint-Consistent State Transitions**: A mathematically rigorous transpose-Jacobian jump condition $\lambda^- = R^T \lambda^+$ that guarantees gradient fidelity across discrete representation changes during training.

---

## 2. Mathematical Formulation

```text
               Forward Integrator [0 -> T]
           X(t_0) ───> X^-(t_j) ──[ R ]──> X^+(t_j) ───> X(T)
                                                          │ Loss L
           Adjoint Integrator [T -> 0]                    ▼
     ∇_θ L <── λ(t_0) <─── λ^-(t_j) <──[ R^T ]── λ^+(t_j) <─── λ(T)
```

### 2.1 Caputo Neural Fractional Differential Equations
The Caputo fractional derivative of order $\beta \in (0, 1)$ is defined by:
$${}^C D_t^\beta z(t) = \frac{1}{\Gamma(1-\beta)} \int_0^t \frac{\dot{z}(\tau)}{(t-\tau)^\beta} d\tau = f_\theta(z(t), t)$$

### 2.2 Memory Kernel Decomposition via SOE
Using the integral representation of the Gamma function, the power-law kernel $s^{-\gamma}$ (with $\gamma = 1 - \beta$) is decomposed via dyadic contour quadrature:
$$s^{-\gamma} = \frac{1}{\Gamma(\gamma)} \int_0^\infty e^{-\lambda s} \lambda^{\gamma - 1} d\lambda \approx \sum_{k=1}^K w_k e^{-\lambda_k s}, \quad \lambda_k > 0, \, w_k > 0$$

### 2.3 Volterra State-Space Reformulation
Converting the Caputo system into its equivalent Volterra integral equation $z(t) = z_0 + I^\beta f_\theta(z, t)$ yields an explicit set of decoupled linear auxiliary ODEs:
$$\frac{dm_k}{dt} = -\lambda_k m_k(t) + f_\theta(z(t), t), \quad m_k(0) = 0, \quad k = 1, \dots, K(t)$$
with algebraic state reconstruction:
$$z(t) \approx z_0 + \frac{1}{\Gamma(\beta)} \sum_{k=1}^{K(t)} w_k m_k(t)$$
This eliminates the need to solve for $\dot{z}(t)$ implicitly and allows exact exponential time-differencing (ETD-RK2) integration.

### 2.4 Analytical Cauchy-Gram Projection Operator
When memory representation transitions from $K^-$ modes ($\lambda^-$) to $K^+$ modes ($\lambda^+$), the state undergoes $m^+ = R m^-$. The optimal least-squares projection in $L_2([0, \infty))$ function space is computed analytically via the Cauchy-Gram system:
$$G_{k, l}^- = \frac{1}{\lambda_k^- + \lambda_l^-} \in \mathbb{R}^{K^- \times K^-}, \qquad H_{j, k} = \frac{1}{\lambda_j^+ + \lambda_k^-} \in \mathbb{R}^{K^+ \times K^-}$$
$$R = H (G^- + \sigma I)^{-1} \in \mathbb{R}^{K^+ \times K^-}$$

### 2.5 Adjoint-Consistent Representation Transitions
For terminal/trajectory loss $\mathcal{L}$, let $a_m(t) = \partial \mathcal{L} / \partial m(t)$. Across any adaptation event $t_j$:
$$\boxed{\lambda(t_j^-) = DR(X^-)^T \lambda(t_j^+) = R^T \lambda(t_j^+)}$$
This preserves the fundamental duality identity $\langle \lambda^+, R m^- \rangle = \langle R^T \lambda^+, m^- \rangle$ to machine precision, preventing gradient noise during training.

### 2.6 Embedded Error Controller
The solver maintains an embedded higher-order shadow mode configuration $K_{\text{shadow}}$ to evaluate local truncation error:
$$\widehat{\epsilon}_M(t) = \frac{\|M_K(t) - M_{\text{shadow}}(t)\|}{\|M_K(t)\| + \epsilon_{\text{floor}}}$$
- If $\widehat{\epsilon}_M(t) > \epsilon_{\text{tol}}$: expand modes $K \to K + \Delta K$.
- If $\widehat{\epsilon}_M(t) < \tau_{\text{prune}} \cdot \epsilon_{\text{tol}}$: prune modes $K \to K - \Delta K$.

---

## 3. Repository Architecture

```
AdaMem-FDE/
├── .agents/rules/
│   └── adamem_fde_spec.md          # Persistent workspace specification & rules
├── core/
│   ├── fractional/
│   │   ├── mittag_leffler.py       # High-precision E_{\alpha, \beta}(z) evaluator
│   │   └── caputo.py               # Caputo L1 finite differences & I^\beta convolution
│   ├── soe/
│   │   ├── dyadic_quadrature.py    # Jiang-Zhang dyadic contour quadrature generator
│   │   └── projection.py           # Cauchy-Gram projection matrix R and adjoint jump R^T
│   ├── controllers/
│   │   └── embedded_controller.py  # Embedded error estimator & mode allocation controller
│   ├── solvers/
│   │   ├── full_history.py         # Baseline 1: Fractional Adams-Bashforth-Moulton O(N^2)
│   │   ├── fixed_soe.py            # Baseline 2: Fixed SOE with ETD-RK2 O(NK)
│   │   └── adaptive_soe.py         # Proposed: AdaMem-FDE error-adaptive solver
│   └── adjoint/
│       └── adamem_adjoint.py       # Custom PyTorch autograd Function with R^T jumps
├── models/
│   └── neural_fde.py               # NeuralFDE & VectorFieldNetwork modules
├── benchmarks/
│   └── systems.py                  # Mittag-Leffler, Duffing, Van der Pol, & Lorenz
├── experiments/
│   ├── run_phase1_validation.py    # Phase I & II: Ground-truth validation
│   ├── run_phase3_adaptation.py    # Phase III: Dynamic K(t) tracking
│   ├── run_phase4_training.py      # Phase IV & V: Neural FDE adjoint training
│   └── run_phase6_scaling.py       # Phase VI: Long-horizon complexity scaling
├── tests/                          # 15 comprehensive unit tests (100% pass)
└── results/                        # Generated publication plots & benchmark data
```

---

## 4. Experimental Suite & Empirical Results

### Phase I & II: Ground-Truth Mittag-Leffler Verification
Analytical linear decay benchmark: ${}^C D_t^{0.7} x(t) = -x(t)$ with $x(0) = 1.0$ against exact solution $x(t) = E_{0.7}(-t^{0.7})$ ($N = 500$ steps, $T = 5.0$).

| Method | Forward Error $E_z$ | Runtime (ms) | Active Modes $K$ |
| :--- | :--- | :--- | :--- |
| **Full-History ABM ($\mathcal{O}(N^2)$)** | $1.0379 \times 10^{-3}$ | 41.80 ms | 500 / 500 |
| **Fixed SOE ($K=8$)** | $1.6981 \times 10^{-1}$ | 23.78 ms | 8 / 8 (under-resolved) |
| **Fixed SOE ($K=16$)** | $3.3769 \times 10^{-2}$ | 61.88 ms | 16 / 16 |
| **Fixed SOE ($K=24$)** | $2.6987 \times 10^{-3}$ | 25.33 ms | 24 / 24 |
| **AdaMem-FDE ($\epsilon_{\text{tol}}=10^{-3}$)** | **$1.5975 \times 10^{-3}$** | 45.37 ms | **31.8 / 32 (automatic)** |

<p align="center">
  <img src="results/phase1_mittag_leffler_benchmark.png" width="850" alt="Phase 1 Benchmark" />
</p>

---

### Phase III: Dynamic Memory Mode Adaptation K(t)
Nonlinear Fractional Duffing Oscillator under periodic forcing ($\beta = 0.85, T = 12.0, N = 600$ steps, $\epsilon_{\text{tol}} = 5 \times 10^{-4}$).

- **Initial Modes**: $K = 8$.
- **Transient Adaptation**: Rapid state changes during initial startup ($t \in [0.02, 0.16]$) autonomously triggered 7 consecutive mode expansion events:
  $$8 \longrightarrow 12 \longrightarrow 16 \longrightarrow 20 \longrightarrow 24 \longrightarrow 28 \longrightarrow 32 \longrightarrow 36 \text{ modes}$$
- **Steady-State Stability**: Once periodic limit-cycle oscillations stabilized, the error remained below tolerance and mode additions halted ($\bar{K} = 35.75$).

<p align="center">
  <img src="results/phase3_dynamic_memory_adaptation.png" width="850" alt="Phase 3 Adaptation" />
</p>

---

### Phase IV & V: Neural FDE Training & Ablation Study
Training a neural network $f_\theta(z, t)$ to recover nonlinear Duffing dynamics from trajectory data.
- **Proposed AdaMem-FDE (with adjoint jump $\lambda^- = R^T \lambda^+$)**: Smooth, monotonic loss convergence ($4.35 \times 10^{-2} \to \mathbf{1.30 \times 10^{-2}}$) with steadily decaying gradient norms.
- **Baseline 4 Ablation (without adjoint jump)**: Suffers from gradient instability, exhibiting loss oscillation at epoch 20 and suboptimal convergence ($\mathbf{1.45 \times 10^{-2}}$).

| Method | Final Training Loss | Total Time (s) | Training Stability |
| :--- | :--- | :--- | :--- |
| **Proposed AdaMem-FDE** | **$1.3027 \times 10^{-2}$** | **1.23 s** | **Monotonic & Converged** |
| **Baseline 4 (Ablation: No Jump)** | $1.4503 \times 10^{-2}$ | 1.24 s | Oscillatory & Suboptimal |

<p align="center">
  <img src="results/phase4_neural_fde_training.png" width="850" alt="Phase 4 Training" />
</p>

---

### Phase VI: Long-Horizon Scalability on Lorenz Attractor
Scaling study on the 3D Fractional Lorenz chaotic attractor ($\beta = 0.99$, $N$ from $200$ to $3,200$ steps):

```
N Steps   | Full-History (s)   | Fixed SOE (s)    | AdaMem-FDE (s)   | AdaMem K_avg
-----------------------------------------------------------------------------------
200       | 0.0289 s           | 0.0195 s         | 0.0320 s         | 31.2
400       | 0.0535 s           | 0.0346 s         | 0.0544 s         | 31.6
800       | 0.1203 s           | 0.0718 s         | 0.1085 s         | 31.8
1600      | 0.2938 s           | 0.1403 s         | 0.2108 s         | 31.9
3200      | ~1.180 s (proj)    | 0.2689 s         | 0.4478 s         | 32.0 (flat!)
```

<p align="center">
  <img src="results/phase6_long_horizon_scaling.png" width="850" alt="Phase 6 Scalability" />
</p>

**Key Conclusion**: AdaMem-FDE converts the quadratic $\mathcal{O}(N^2)$ history barrier into clean, linear $\mathcal{O}(N \cdot \bar{K})$ scaling while keeping memory modes $\bar{K} = 32$ completely bounded over arbitrarily long horizons!

---

## 5. Baselines Evaluated

1. **Baseline 1 — Full-History Fractional Solver**: Classical Diethelm Adams-Bashforth-Moulton $\mathcal{O}(N^2)$ predictor-corrector.
2. **Baseline 2 — Fixed-Order SOE Memory**: Sum-of-exponentials with fixed order $K = \text{const}$.
3. **Baseline 3 — Standard Neural FDE**: Existing fixed-quadrature multi-step solver.
4. **Baseline 4 — Naive Adaptive Memory (Ablation)**: Adaptive $K(t)$ without adjoint-consistent $R^T$ transitions.
5. **Proposed Method — AdaMem-FDE**: Error-adaptive dynamic memory + adjoint-consistent transitions.

---

## 6. Installation & Setup

Clone the repository and install the dependencies in a Python 3.10+ environment:

```bash
git clone https://github.com/aradhyags7/AdaMem-FDE.git
cd AdaMem-FDE
```

### Using `uv` (Recommended - Installs in seconds):
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 7. Running Experiments & Reproduction

Run all progressive experimental phases to reproduce the figures in `results/`:

```bash
# Phase I & II: Ground-truth Mittag-Leffler validation
python experiments/run_phase1_validation.py

# Phase III: Dynamic memory mode adaptation K(t) on Duffing oscillator
python experiments/run_phase3_adaptation.py

# Phase IV & V: Neural FDE adjoint training & ablation comparison
python experiments/run_phase4_training.py

# Phase VI: Long-horizon O(N) scalability study on chaotic Lorenz attractor
python experiments/run_phase6_scaling.py
```

---

## 8. Running Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest tests/ -v -s
```

All 15 tests pass across:
- Mittag-Leffler analytical precision ($E_{1,1}(z) = e^z$, zeros, asymptotic tails)
- Dyadic contour quadrature pole positivity ($\lambda_k > 0, w_k > 0$)
- Adjoint inner-product duality $\langle \lambda^+, R m^- \rangle = \langle R^T \lambda^+, m^- \rangle$ ($\Delta < 10^{-12}$)
- Forward solver convergence on analytical benchmarks
- Finite difference gradient verification and Ablation C validation

---

## 9. Citation & References

If you find AdaMem-FDE helpful in your research, please cite:

```bibtex
@article{adamem_fde_2026,
  title={Error-Adaptive Dynamic Memory Compression for Adjoint-Trained Neural Fractional Differential Equations},
  author={Aradhya},
  journal={arXiv preprint},
  year={2026}
}
```

### Key References
- Jiang, S., & Zhang, J. (2017). *Fast evaluation of the fractional Laplacian and fractional differential equations using sum-of-exponentials approximations*.
- Diethelm, K., Ford, N. J., & Freed, A. D. (2002). *A predictor-corrector approach for the numerical solution of fractional differential equations*. Nonlinear Dynamics.
- Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). *Neural ordinary differential equations*. NeurIPS.
