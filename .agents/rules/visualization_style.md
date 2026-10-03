# Visualization Style & Theme Invariant: Publication Light Mode Only

All benchmark graphs, experimental diagnostic plots, convergence curves, and scientific figures generated in this repository must **ALWAYS** be rendered in **Light Mode**. Dark mode figures are strictly prohibited.

## Core Invariants
1. **Never Dark Mode**: Never use dark backgrounds (e.g., `#0f172a`, `#1e293b`, `black`, `dark_background`, or dark slate/navy themes) for figure canvases, subplots, or legends.
2. **Pure White Background**: Every figure canvas and subplot axis must use pure white (`#ffffff` or `"white"`):
   ```python
   fig.patch.set_facecolor("white")
   ax.set_facecolor("white")
   ```
3. **High-Contrast Dark Typography**: All titles, axis labels, tick labels, and annotations must use dark colors (`#0f172a`, `#1e293b`, or `"black"`) for maximum contrast and readability in LaTeX documents, arXiv submissions, and physical printouts.
4. **Clean Subtle Grids**: Use light gray grid lines:
   ```python
   ax.grid(True, which="both", ls=":", color="#cbd5e1", alpha=0.6)
   ax.tick_params(colors="#1e293b", which="both")
   ```
5. **Crisp Legend Styling**: Legends must have white backgrounds with light borders and dark text:
   ```python
   ax.legend(facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)
   ```
6. **Publication Export**: Always save figures at $\ge 300$ DPI with `bbox_inches="tight"` and `facecolor="white"`:
   ```python
   plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
   ```
7. **Dual Archival**: Any figure used in the paper must be saved in `results/` and mirrored to `paper/figures/`.
