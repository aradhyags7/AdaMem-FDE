# AdaMem-FDE: Core Research Invariants & Architectural Guidelines

## 1. Mathematical Formulation
- **System**: Caputo Neural Fractional Differential Equation
  $${}^C D_t^\beta z(t) = f_\theta(z(t), t), \quad 0 < \beta \le 1$$
- **Volterra Integral Representation**:
  Auxiliary memory states satisfy:
  $$\dot{m}_k(t) = -\lambda_k m_k(t) + f_\theta(z(t), t), \quad k = 1, \dots, K(t)$$
  with state reconstruction:
  $$z(t) \approx z_0 + \frac{1}{\Gamma(\beta)} \sum_{k=1}^{K(t)} w_k m_k(t)$$
  Auxiliary states are initialized at $m_k(0) = 0$.

## 2. Dynamic Memory Transitions & Adjoint Consistency
- **State Transition Map**:
  When memory representation adapts ($K^- \to K^+$), the state undergoes:
  $$X^+ = R(X^-) \implies m^+ = R m^-$$
  where $R$ is a linear projection operator derived from least-squares kernel matching.
- **Adjoint Jump Condition**:
  At any representation transition event $t_j$:
  $$\lambda(t_j^-) = DR(X^-)^T \lambda(t_j^+) = R^T \lambda(t_j^+)$$
  Adjoints must NEVER be discarded or naively zero-initialized without applying the transpose Jacobian transition.

## 3. Error-Controlled Memory Adaptation
- **Estimator**: Embedded SOE pair computing local error $\widehat{\epsilon}_M(t) = \|z_K(t) - z_{high}(t)\|$.
- **Trigger**:
  - If $\widehat{\epsilon}_M(t) > \epsilon_{tol}$: expand representation (mode insertion / reconfiguration).
  - If $\widehat{\epsilon}_M(t) < \tau_{prune} \cdot \epsilon_{tol}$: prune redundant modes or merge.

## 4. Verification & Validation Protocols
- **Phase I Benchmark**: Analytical Mittag-Leffler solution $x(t) = x_0 E_\beta(-\lambda t^\beta)$ for ${}^C D_t^\beta x = -\lambda x$.
- **Gradient Verification**: Adjoint gradients $g_{\text{adj}} = \nabla_\theta \mathcal{L}$ must be systematically validated against two-sided finite differences:
  $$E_g = \frac{\|\nabla_\theta \mathcal{L}_{\text{adj}} - \nabla_\theta \mathcal{L}_{\text{FD}}\|}{\max(\|\nabla_\theta \mathcal{L}_{\text{FD}}\|, 10^{-7})} < 10^{-4}$$
- **Required Baselines**:
  1. Full-history fractional solver ($\mathcal{O}(N^2)$ reference)
  2. Fixed-order SOE memory ($K = \text{const}$)
  3. Existing Neural FDE benchmark (e.g., torchfde / Adams-Bashforth-Moulton)
  4. Naive adaptive memory without adjoint-consistent transitions (ablation baseline)
  5. **AdaMem-FDE (Proposed)**: Adaptive memory + adjoint-consistent transitions.

## 5. Benchmark Archiving Protocol
- Before running any new benchmark, always move existing PNG graphs and JSON/CSV data from `results/` into `previous_results/<experiment_run_name>/` (see `.agents/rules/results_archiving.md`).
- `results/` must strictly hold the canonical headline figures referenced in `README.md`.

## 6. Metrics to Report
- Forward error $E_z = \|z - z_{\text{ref}}\| / \|z_{\text{ref}}\|$
- Relative gradient error $E_g$
- Wall-clock runtime $T_{\text{runtime}}$
- Peak GPU/RAM memory $M_{\text{peak}}$
- Average active modes $\bar{K} = \frac{1}{N} \sum_{n=1}^N K_n$ and maximum modes $K_{\max}$
- Adaptation frequency $N_{\text{adapt}}$
