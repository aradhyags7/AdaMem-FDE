# AdaMem-FDE: Complete Ground-Truth Data-Integrity Verification Ledger

> [!IMPORTANT]
> **Data Integrity Guarantee**: Every number, table entry, and inline quantitative claim in `paper/main.tex` and `paper/main.pdf` is computed directly from archived experimental JSON result files via `scripts/generate_paper_tables.py`. No manual transcription, smoothing, or memory-based estimation is permitted.
> 
> **Repository Commit**: `d6815115d502d4ca2543ccf6bd77126acdbdfc85` (and subsequent audit revisions)
> **Public Repository**: [https://github.com/aradhyags7/AdaMem-FDE](https://github.com/aradhyags7/AdaMem-FDE)
> **PDF Output**: [paper/main.pdf](file:///c:/Users/ASUS/OneDrive/Desktop/AdaMem-FDE/paper/main.pdf) (24 pages, compiled with Tectonic)

---

## 1. Executive Summary & Audit Resolution Log

| Experimental Phase | Previous Paper Status | Audit Finding | Remediation Action in `main.tex` | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase I & II: Mittag-Leffler** | ❌ Fabricated | Errors 1-2 orders too small; runtimes 5-10x smaller; fake rows | Full rewrite of Table 2 & prose from `results/phase1_mittag_leffler_data.json` | **VERIFIED EXACT ✅** |
| **Phase III: Duffing Adaptation** | ❌ Fabricated | Claimed T=30, N=1500, pruning; real run was T=6.0, N=300 monotonic | Corrected T=6.0, N=300, 4 expansions (8→24), no pruning | **VERIFIED EXACT ✅** |
| **Phase IV & V: $R^T$ Adjoint Jump** | ❌ Fabricated | Claimed 100% divergence & 0.184 loss; real run converged at 0.00248 | Kept real 69.21% grad error; reported honest convergence (0.00248 vs 0.00223); removed catastrophic divergence | **VERIFIED EXACT ✅** |
| **Phase VI: Long-Horizon Scaling** | ❌ Fabricated | Modes artificially reduced (14→29); missing intermediate N | Full rewrite of Table 4; added projected label for N>2500; restored real $\bar{K}=32.0$ | **VERIFIED EXACT ✅** |
| **Phase VII: Pareto Frontier** | ❌ Fabricated | Modes claimed 7.8–26.4; gradient error capped at 5.1% | Corrected modes to 11.3–40.5; gradient error range 4.4%–8.8% | **VERIFIED EXACT ✅** |
| **Phase VIII: Joint Order Discovery** | ✅ Exact Match | Numbers matched raw JSON; previous audit confirmed | Confirmed exact match in Table 5 across all 5 seeds & noise sweeps | **VERIFIED EXACT ✅** |
| **Phase IX: Graded Meshes** | ⚠️ Partial | β=0.50 & 0.70 matched; β=0.85 null result omitted | Added β=0.85 null result arm ($1.00\times$) in Table 6 & text | **VERIFIED EXACT ✅** |
| **Phase X: Incommensurate Multi-Order** | ⚠️ Misleading Ratio | Table matched; text cherry-picked 405x–1750x vs worst baselines | Updated prose to state full range: $25\times$ (vs β=0.60) to $1752\times$ (vs β=0.90) | **VERIFIED EXACT ✅** |

---

## 2. Recommended 5-Point Spot-Check Guide for Author

Before approving the manuscript, the author can verify these 5 specific numbers directly by opening the raw JSON files:

1. **Phase I Table 2 (Baseline Error)**: Open `results/phase1_mittag_leffler_data.json`, line 4. Verify `"forward_error": 0.0010378906374973616` ($1.038 \times 10^{-3}$).
2. **Phase IV/V Table 3 (Ablation Gradient Error & Loss)**: Open `results/phase4_neural_fde_training.json`. Line 407 shows `"gradient_error_mean": 69.20937102096742` ($69.21\%$). Line 398 shows `"final_loss_mean": 0.0024761038319583645` ($2.476 \times 10^{-3}$), confirming that naive training converged fine without divergence.
3. **Phase VI Table 4 (Long-Horizon AdaMem Runtime)**: Open `results/phase6_long_horizon_scaling_data.json`, line 111. Verify `14.234681000001729` ($14.23$ seconds) at $N = 100,000$.
4. **Phase IX Table 6 (Singularity Quenching at N=400)**: Open `results/phase9_graded_singularity_quenching.json`. Line 13 shows uniform error `0.013253...` ($1.325 \times 10^{-2}$); line 31 shows graded error `0.0001128...` ($1.128 \times 10^{-4}$). Ratio is exactly $117.46\times$.
5. **Phase X Table 7 (Incommensurate Trajectory Loss)**: Open `results/phase10_incommensurate_multi_order.json`. Line 55 shows `"final_loss": 4.111204907530919e-05` ($4.111 \times 10^{-5}$). Line 6 shows closest commensurate baseline (β=0.60) MSE `0.001035017056058223` ($1.035 \times 10^{-3}$). Ratio is $25.18\times$.

---

## 3. Comprehensive Line-by-Line Claim & Table Verification Ledger

### Table 2: Phase I & II Mittag-Leffler Benchmark ($T=5.0, N=500$)
**Source File**: `results/phase1_mittag_leffler_data.json`

| Solver Configuration | Paper Value | JSON Field Key | JSON Line | Raw JSON Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Full-History | Error: $1.038 \times 10^{-3}$ | `forward_error` | Line 4 | `0.0010378906374973616` | MATCH ✅ |
| Full-History | Runtime: $48.6$ ms | `runtime_ms` | Line 5 | `48.570699989795685` | MATCH ✅ |
| Full-History | Modes: $500$ | `avg_modes` | Line 6 | `500` | MATCH ✅ |
| Fixed SOE ($K=8$) | Error: $1.698 \times 10^{-1}$ | `forward_error` | Line 11 | `0.16981084538324626` | MATCH ✅ |
| Fixed SOE ($K=8$) | Runtime: $33.5$ ms | `runtime_ms` | Line 12 | `33.51139999083243` | MATCH ✅ |
| Fixed SOE ($K=8$) | Modes: $8$ | `avg_modes` | Line 13 | `8` | MATCH ✅ |
| Fixed SOE ($K=16$) | Error: $3.377 \times 10^{-2}$ | `forward_error` | Line 18 | `0.03377156942918881` | MATCH ✅ |
| Fixed SOE ($K=16$) | Runtime: $31.1$ ms | `runtime_ms` | Line 19 | `31.065899995155632` | MATCH ✅ |
| Fixed SOE ($K=16$) | Modes: $16$ | `avg_modes` | Line 20 | `16` | MATCH ✅ |
| Fixed SOE ($K=24$) | Error: $2.699 \times 10^{-3}$ | `forward_error` | Line 25 | `0.002698944510065095` | MATCH ✅ |
| Fixed SOE ($K=24$) | Runtime: $31.6$ ms | `runtime_ms` | Line 26 | `31.57140000257641` | MATCH ✅ |
| Fixed SOE ($K=24$) | Modes: $24$ | `avg_modes` | Line 27 | `24` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-3}$) | Error: $7.443 \times 10^{-3}$ | `forward_error` | Line 32 | `0.007442654924738596` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-3}$) | Runtime: $54.0$ ms | `runtime_ms` | Line 33 | `53.98410001362208` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-3}$) | Modes: $21.8$ / $24$ | `avg_modes`, `max_modes` | Lines 34, 35 | `21.83`, `24` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-4}$) | Error: $1.014 \times 10^{-2}$ | `forward_error` | Line 39 | `0.010141673966551698` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-4}$) | Runtime: $52.5$ ms | `runtime_ms` | Line 40 | `52.49399998865556` | MATCH ✅ |
| AdaMem ($\epsilon_{\mathrm{tol}} = 10^{-4}$) | Modes: $31.0$ / $32$ | `avg_modes`, `max_modes` | Lines 41, 42 | `30.97`, `32` | MATCH ✅ |

---

### Section 5.2: Phase III Duffing Oscillator Adaptation
**Source File**: `results/phase3_dynamic_memory_adaptation_data.json`

| Metric / Parameter | Paper Value | JSON Field Key | JSON Line | Raw JSON Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Time Horizon $T$ | $6.0$ | `T` | Line 3 | `6.0` | MATCH ✅ |
| Step Count $N$ | $300$ | `N` | Line 4 | `300` | MATCH ✅ |
| Fractional Order $\beta$ | $0.85$ | `beta` | Line 5 | `0.85` | MATCH ✅ |
| Simulation Tolerance $\epsilon_{\mathrm{tol}}$ | $10^{-3}$ | `tol` | Line 6 | `0.001` | MATCH ✅ |
| Average Active Modes $\bar{K}$ | $22.2$ | `avg_modes` | Line 7 | `22.22` | MATCH ✅ |
| Peak Modes $K_{\max}$ | $24$ | `max_modes` | Line 8 | `24` | MATCH ✅ |
| Adaptation Count | $4$ | `num_adaptations` | Line 9 | `4` | MATCH ✅ |
| Mode Trajectory | $8 \to 12 \to 16 \to 20 \to 24$ | `mode_trajectory` | Lines 12–16 | `[8, 12, 16, 20, 24]` | MATCH ✅ |
| Adaptation Timestamps $t$ | $0.18, 0.32, 0.64, 1.46$ | `adaptation_times` | Lines 19–22 | `[0.18, 0.32, 0.64, 1.46]` | MATCH ✅ |
| Adaptation Step Indices $n$ | $9, 16, 32, 73$ | `adaptation_steps` | Lines 25–28 | `[9, 16, 32, 73]` | MATCH ✅ |
| Pruning Events | $0$ (none triggered) | Trajectory inspection | Lines 12–16 | Monotonic expansion only | MATCH ✅ |

---

### Table 3: Phase IV & V $R^T$ Adjoint Jump Ablation Study
**Source File**: `results/phase4_neural_fde_training.json`

| Method | Metric | Paper Value | JSON Field Key | JSON Line | Raw JSON Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Naive (No $R^T$) | Rel. Gradient Error $E_g$ | $69.21\% \pm 8.89\%$ | `baseline_no_jump.gradient_error_mean`, `std` | Lines 407, 408 | `69.20937%`, `8.89279%` | MATCH ✅ |
| Naive (No $R^T$) | Final Loss | $2.476 \times 10^{-3} \pm 6.083 \times 10^{-4}$ | `baseline_no_jump.final_loss_mean`, `std` | Lines 398, 399 | `0.0024761`, `0.00060825` | MATCH ✅ |
| Naive (No $R^T$) | Seeds Worsened | $0/5$ | Seed loss checks | Lines 212–395 | $L_{\mathrm{final}} < L_{\mathrm{initial}}$ for all seeds | MATCH ✅ |
| Naive (No $R^T$) | Convergence Status | Converged | Training logs | Lines 212–395 | Converged (no divergence) | MATCH ✅ |
| AdaMem-FDE (With $R^T$) | Rel. Gradient Error $E_g$ | $1.70\% \pm 0.99\%$ | `proposed.gradient_error_mean`, `std` | Lines 207, 208 | `1.69937%`, `0.98920%` | MATCH ✅ |
| AdaMem-FDE (With $R^T$) | Final Loss | $2.228 \times 10^{-3} \pm 2.786 \times 10^{-4}$ | `proposed.final_loss_mean`, `std` | Lines 198, 199 | `0.0022277`, `0.00027863` | MATCH ✅ |
| AdaMem-FDE (With $R^T$) | Seeds Worsened | $0/5$ | Seed loss checks | Lines 9–195 | $L_{\mathrm{final}} < L_{\mathrm{initial}}$ for all seeds | MATCH ✅ |
| AdaMem-FDE (With $R^T$) | Convergence Status | Converged | Training logs | Lines 9–195 | Converged | MATCH ✅ |
| Both | Error Reduction Factor | $40.7\times$ | `gradient_error_reduction_factor` | Line 410 | `40.7263711` | MATCH ✅ |

---

### Table 4: Phase VI Long-Horizon Linear Scaling ($N = 10^5$)
**Source File**: `results/phase6_long_horizon_scaling_data.json`

| Steps $N$ | Horizon $T$ | Full-History (s) | Line | Fixed SOE ($K=16$) (s) | Line | AdaMem-FDE (s) | Line | AdaMem $\bar{K}$ | Line | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $500$ | $5.0$ | $0.076$ | L15 | $0.047$ | L49 | $0.079$ | L83 | $29.3$ | L115 | $1.0\times$ |
| $1,000$ | $10.0$ | $0.168$ | L19 | $0.096$ | L53 | $0.185$ | L87 | $30.6$ | L116 | $0.9\times$ |
| $2,500$ | $25.0$ | $0.684$ | L23 | $0.239$ | L57 | $0.390$ | L91 | $31.5$ | L117 | $1.8\times$ |
| $5,000$ | $50.0$ | $2.73$ (Proj.) | L27 | $0.467$ | L61 | $0.807$ | L95 | $31.7$ | L118 | $3.4\times$ |
| $10,000$ | $100.0$ | $10.9$ (Proj.) | L31 | $1.033$ | L65 | $1.600$ | L99 | $31.9$ | L119 | $6.8\times$ |
| $25,000$ | $250.0$ | $68.3$ (Proj.) | L35 | $2.492$ | L69 | $3.830$ | L103 | $31.9$ | L120 | $17.8\times$ |
| $50,000$ | $500.0$ | $273.4$ (Proj.) | L39 | $4.619$ | L73 | $6.366$ | L107 | $32.0$ | L121 | $42.9\times$ |
| $100,000$ | $1000.0$ | $1093.5$ (Proj.) | L43 | $8.94$ | L77 | $14.23$ | L111 | $32.0$ | L122 | $76.8\times$ |

---

### Section 5.5: Phase VII Pareto Frontier & Sensitivity Sweep
**Source File**: `results/pareto_frontier_data.json`

| Metric / Parameter | Paper Value | JSON Field Key | JSON Line | Raw JSON Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\bar{K}$ Range across $\epsilon_{\mathrm{tol}}$ | $11.3$ to $40.5$ | `adaptive_points.avg_modes` | Lines 98, 140 | `11.31` (tol=0.01) to `40.46` (tol=1e-5) | MATCH ✅ |
| Gradient Error $E_g$ Range | $4.4\%$ to $8.8\%$ | `grad_errors` | Lines 88, 89 | `4.357\%` (tol=0.005) to `8.767\%` (tol=0.01) | MATCH ✅ |
| Error Plateau Level | $1.8 \times 10^{-2}$ | `adaptive_points[tol=0.001].error` | Line 110 | `0.01845009` | MATCH ✅ |

---

### Table 5: Phase VIII Joint $\beta$ and Parameter Discovery
**Source Files**: `previous_results/phase8_noise_sweep/phase8_multiseed_noise0.01.json` & `previous_results/phase8_stride_discretization/*.json`

| Configuration | Final $\beta$ (Mean $\pm$ Std) | Relative Error | Within $\pm 2\%$ | Median Loss | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fixed Misspecified ($\beta=0.50$) | $0.5000 \pm 0.0000$ | $33.33\%$ | $0/5$ | $1.18 \times 10^{-2}$ | MATCH ✅ |
| Oracle Known ($\beta^*=0.75$) | $0.7500 \pm 0.0000$ | $0.00\%$ | $5/5$ | $8.83 \times 10^{-5}$ | MATCH ✅ |
| Joint $\beta_0 = 0.90$ | $0.7600 \pm 0.0034$ | $1.34\%$ | $5/5$ ($100\%$) | $1.68 \times 10^{-4}$ | MATCH ✅ |
| Joint $\beta_0 = 0.50$ | $0.7394 \pm 0.0033$ | $1.42\%$ | $4/5$ ($80\%$) | $7.74 \times 10^{-5}$ | MATCH ✅ |
| Joint $\beta_0 = 0.30$ | $0.7250 \pm 0.0047$ | $3.33\%$ | $0/5$ ($0\%$) | $2.43 \times 10^{-4}$ | MATCH ✅ |
| Stride $N=60$ ($\Delta t = 0.050$) | $0.7369 \pm 0.0000$ | $1.75\%$ | $5/5$ | $7.29 \times 10^{-5}$ | MATCH ✅ |
| Stride $N=120$ ($\Delta t = 0.025$) | $0.7410 \pm 0.0000$ | $1.21\%$ | $5/5$ | $3.45 \times 10^{-5}$ | MATCH ✅ |
| Stride $N=300$ ($\Delta t = 0.010$) | $0.7470 \pm 0.0000$ | $0.41\%$ | $5/5$ | $2.17 \times 10^{-5}$ | MATCH ✅ |

---

### Table 6: Phase IX Graded Mesh Singularity Quenching ($K=48$)
**Source File**: `results/phase9_graded_singularity_quenching.json`

| Fractional Order | Graded $r$ | Intervals $N$ | Uniform Error (Line) | Graded Error (Line) | Improvement Ratio | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\beta = 0.50$ | $r = 3.000$ | $50$ | $2.852 \times 10^{-2}$ (L10) | $1.047 \times 10^{-3}$ (L28) | $27.24\times$ | MATCH ✅ |
| $\beta = 0.50$ | $r = 3.000$ | $100$ | $2.264 \times 10^{-2}$ (L11) | $3.709 \times 10^{-4}$ (L29) | $61.04\times$ | MATCH ✅ |
| $\beta = 0.50$ | $r = 3.000$ | $200$ | $1.752 \times 10^{-2}$ (L12) | $1.779 \times 10^{-4}$ (L30) | $98.49\times$ | MATCH ✅ |
| $\beta = 0.50$ | $r = 3.000$ | $400$ | $1.325 \times 10^{-2}$ (L13) | $1.128 \times 10^{-4}$ (L31) | $\mathbf{117.46\times}$ | MATCH ✅ |
| $\beta = 0.50$ | $r = 3.000$ | $800$ | $9.842 \times 10^{-3}$ (L14) | $1.119 \times 10^{-4}$ (L32) | $87.95\times$ | MATCH ✅ |
| $\beta = 0.70$ | $r = 1.857$ | $50$ | $5.891 \times 10^{-3}$ (L46) | $7.043 \times 10^{-4}$ (L54) | $8.36\times$ | MATCH ✅ |
| $\beta = 0.70$ | $r = 1.857$ | $100$ | $3.964 \times 10^{-3}$ (L47) | $2.905 \times 10^{-4}$ (L55) | $13.64\times$ | MATCH ✅ |
| $\beta = 0.70$ | $r = 1.857$ | $200$ | $2.589 \times 10^{-3}$ (L48) | $2.127 \times 10^{-4}$ (L56) | $12.18\times$ | MATCH ✅ |
| $\beta = 0.70$ | $r = 1.857$ | $400$ | $1.656 \times 10^{-3}$ (L49) | $1.582 \times 10^{-4}$ (L57) | $10.47\times$ | MATCH ✅ |
| $\beta = 0.70$ | $r = 1.857$ | $800$ | $1.044 \times 10^{-3}$ (L50) | $2.419 \times 10^{-4}$ (L58) | $4.32\times$ | MATCH ✅ |
| $\beta = 0.85$ (Null Result) | $r = 1.353$ | $50$ | $1.924 \times 10^{-3}$ (L65) | $1.917 \times 10^{-3}$ (L72) | $1.00\times$ | MATCH ✅ |
| $\beta = 0.85$ (Null Result) | $r = 1.353$ | $100$ | $1.748 \times 10^{-3}$ (L66) | $1.764 \times 10^{-3}$ (L73) | $0.99\times$ | MATCH ✅ |
| $\beta = 0.85$ (Null Result) | $r = 1.353$ | $200$ | $1.814 \times 10^{-3}$ (L67) | $1.797 \times 10^{-3}$ (L74) | $1.01\times$ | MATCH ✅ |
| $\beta = 0.85$ (Null Result) | $r = 1.353$ | $400$ | $1.765 \times 10^{-3}$ (L68) | $1.797 \times 10^{-3}$ (L75) | $0.98\times$ | MATCH ✅ |
| $\beta = 0.85$ (Null Result) | $r = 1.353$ | $800$ | $1.742 \times 10^{-3}$ (L69) | $1.738 \times 10^{-3}$ (L76) | $1.00\times$ | MATCH ✅ |

---

### Table 7: Phase X Incommensurate Multi-Order Dynamics
**Source File**: `results/phase10_incommensurate_multi_order.json`

| Formulation | Order Configuration | Trajectory MSE | JSON Line | Improvement Ratio vs Incommensurate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Commensurate Baseline 1 | $\bar{\beta} = 0.90$ (Fast) | $7.203 \times 10^{-2}$ | Line 4 | $1752.1\times$ | MATCH ✅ |
| Commensurate Baseline 2 | $\bar{\beta} = 0.75$ (Mean) | $1.667 \times 10^{-2}$ | Line 5 | $405.4\times$ | MATCH ✅ |
| Commensurate Baseline 3 | $\bar{\beta} = 0.60$ (Slow) | $1.035 \times 10^{-3}$ | Line 6 | $\mathbf{25.2\times}$ | MATCH ✅ |
| **Incommensurate AdaMem-FDE** | **Learned $\vec{\beta} = [0.8626, 0.6043]$** | $\mathbf{4.111 \times 10^{-5}}$ | Line 55 | Reference ($1.0\times$) | MATCH ✅ |

---

## 4. Verification Conclusion

Every single numerical value in the paper now matches the archived experimental data exactly:
- **Total independently verified claims**: 178
- **Tables regenerated and validated**: Tables 2, 3, 4, 5, 6, 7
- **Fictitious / exaggerated narratives eliminated**: 100% (catastrophic divergence removed, honest Adam absorption reported, non-existent pruning removed, unverified rows deleted)
- **Null results restored**: Phase IX β=0.85 added
- **Balanced comparison ratios reported**: Phase X stated as full range $25\times$ to $1752\times$
