"""
Rekarisk Image Export — Export plots to image files.

Provides standalone export of consequence analysis plots as PNG, SVG, or PDF
without requiring the full UI. Supports:
    - Contour plots (dispersion, thermal radiation, overpressure)
    - FN curves
    - Risk matrices
    - Batch export of all plots from a project
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless export
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import ScalarFormatter, LogFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Colormaps
# ══════════════════════════════════════════════════════════════════════════════

if HAS_MPL:
    # Custom Rekarisk colormap: white → light blue → blue → orange → red
    _REKARISK_CMAP = LinearSegmentedColormap.from_list(
        "rekarisk", [
            (0.0, "#FFFFFF"),
            (0.2, "#A9CCE3"),
            (0.4, "#2E86C1"),
            (0.6, "#F39C12"),
            (0.8, "#E74C3C"),
            (1.0, "#922B21"),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# Contour Plot
# ══════════════════════════════════════════════════════════════════════════════

def export_contour_plot(
    result: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "png",
    dpi: int = 150,
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
    cmap: Any = None,
) -> str:
    """Export a 2D contour/filled-contour plot from a result dict.

    The result dict should contain one of:
        - grid_data: {"x": [...], "y": [...], "Z": 2D array, "levels": [...]}
        - or: "x", "y", "Z", "levels" directly at top level
        - or: pre-computed contours (list of segments)

    Parameters
    ----------
    result : dict
        Result data with grid and contour info.
    output_path : str or Path
        Where to save the image.
    format : str
        "png", "svg", "pdf", "jpg".
    dpi : int
        Resolution for raster formats.
    title : str
        Plot title. Auto-generated if empty.
    figsize : tuple
        Figure size in inches.
    cmap : colormap, optional
        Colormap for filled contours.

    Returns
    -------
    str
        The output path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for image export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cmap is None:
        cmap = _REKARISK_CMAP

    grid = result.get("grid_data", result)
    x = np.asarray(grid.get("x", []), dtype=float)
    y = np.asarray(grid.get("y", []), dtype=float)
    Z = np.asarray(grid.get("Z", grid.get("values", [])), dtype=float)
    levels = grid.get("levels", [])

    if not title:
        title = result.get("name", result.get("type", "Contour Plot"))

    fig, ax = plt.subplots(figsize=figsize)

    if Z.size > 0 and Z.ndim == 2 and x.size > 0 and y.size > 0:
        if levels:
            # Filled contours with labeled lines
            cf = ax.contourf(x, y, Z, levels=levels, cmap=cmap, extend="both")
            ax.contour(x, y, Z, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
            cbar = fig.colorbar(cf, ax=ax, label="Concentration / Intensity")
        else:
            # Pcolormesh fallback
            cf = ax.pcolormesh(x, y, Z, cmap=cmap, shading="auto")
            cbar = fig.colorbar(cf, ax=ax, label="Value")

        # Source marker
        ax.plot(0, 0, "k*", markersize=12, label="Source")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Downwind Distance (m)")
        ax.set_ylabel("Crosswind Distance (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
    else:
        # Try pre-computed contour segments
        contours = grid.get("contours", grid.get("contour", []))
        if isinstance(contours, list) and len(contours) > 0:
            for i, seg in enumerate(contours):
                if isinstance(seg, (list, np.ndarray)) and len(seg) >= 2:
                    pts = np.asarray(seg)
                    ax.plot(pts[:, 0], pts[:, 1], label=f"Level {i+1}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No contour data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="grey")

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# FN Curve
# ══════════════════════════════════════════════════════════════════════════════

def export_fn_curve(
    fn_data: Union[Dict[str, Any], List[Tuple[float, float]]],
    output_path: Union[str, Path],
    format: str = "png",
    dpi: int = 150,
    title: str = "FN Curve — Societal Risk",
    figsize: Tuple[float, float] = (8, 6),
    criterion: Optional[Dict[str, Any]] = None,
) -> str:
    """Export an FN curve plot.

    Parameters
    ----------
    fn_data : dict or list
        Either {"n": [...], "f": [...]} or list of (n, f) tuples.
    output_path : str or Path
        Save path.
    format : str
        Image format.
    dpi : int
        Resolution.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    criterion : dict, optional
        FN criterion line: {"name": str, "n": [...], "f": [...]} for overlay.

    Returns
    -------
    str
        The output path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for image export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Extract N, F from various formats
    if isinstance(fn_data, dict):
        n_vals = np.asarray(fn_data.get("n", []), dtype=float)
        f_vals = np.asarray(fn_data.get("f", []), dtype=float)
    else:
        n_vals = np.array([row[0] for row in fn_data], dtype=float)
        f_vals = np.array([row[1] for row in fn_data], dtype=float)

    if len(n_vals) > 0:
        ax.loglog(n_vals, f_vals, "o-", color="#E74C3C", linewidth=2, markersize=6,
                  label="Calculated FN curve")

        # Fill area under curve
        ax.fill_between(n_vals, f_vals, alpha=0.15, color="#E74C3C")

    # Criterion line
    if criterion:
        cn = np.asarray(criterion.get("n", []), dtype=float)
        cf = np.asarray(criterion.get("f", []), dtype=float)
        if len(cn) > 0:
            ax.loglog(cn, cf, "--", color="#2E86C1", linewidth=2,
                      label=criterion.get("name", "Criterion"))

    ax.set_xlabel("Number of Fatalities (N)", fontsize=11)
    ax.set_ylabel("Cumulative Frequency (per year)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    # ALARP zone shading (HSE UK)
    if len(n_vals) > 0:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.axhspan(1e-3, ylim[1], xmin=0, xmax=1, color="red", alpha=0.05, label="_unacceptable")
        ax.axhspan(1e-6, 1e-3, xmin=0, xmax=1, color="orange", alpha=0.05, label="_alarp")
        ax.axhspan(ylim[0], 1e-6, xmin=0, xmax=1, color="green", alpha=0.05, label="_broadly acceptable")

        # Region labels
        mid_y = np.sqrt(ylim[0] * 2e-6)
        if mid_y > ylim[0]:
            ax.text(n_vals[-1] * 0.6, 1e-4, "ALARP Region", fontsize=8, color="orange",
                    ha="center", alpha=0.7)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Risk Matrix
# ══════════════════════════════════════════════════════════════════════════════

def export_risk_matrix_image(
    matrix: Union[List[List[int]], Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "png",
    dpi: int = 150,
    title: str = "Risk Matrix",
    figsize: Tuple[float, float] = (8, 7),
) -> str:
    """Export a risk matrix as a styled image.

    Parameters
    ----------
    matrix : list[list[int]] or dict
        Risk matrix data. If list, 2D array of risk levels (0-based).
        If dict, should have "data" (2D) and optionally "row_labels", "col_labels".
    output_path : str or Path
        Save path.
    format : str
        Image format.
    dpi : int
        Resolution.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    str
        The output path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for image export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(matrix, dict):
        data = matrix.get("data", matrix.get("matrix", []))
        row_labels = matrix.get("row_labels", matrix.get("likelihood_labels", []))
        col_labels = matrix.get("col_labels", matrix.get("consequence_labels", []))
    else:
        data = matrix
        row_labels = []
        col_labels = []

    data_arr = np.asarray(data, dtype=float)

    # Default labels
    if not col_labels:
        col_labels = [f"C{i+1}" for i in range(data_arr.shape[1])]
    if not row_labels:
        row_labels = [f"L{data_arr.shape[0] - i}" for i in range(data_arr.shape[0])]

    # Risk colors: green=0, yellow=1, orange=2, red=3, dark red=4+
    risk_colors = [
        "#27AE60",  # 0 — Acceptable
        "#F1C40F",  # 1 — Low
        "#F39C12",  # 2 — Medium
        "#E74C3C",  # 3 — High
        "#922B21",  # 4 — Extreme
    ]
    risk_labels = {
        0: "Acceptable",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Extreme",
    }

    n_rows, n_cols = data_arr.shape

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):
            val = int(data_arr[i, j])
            color = risk_colors[min(val, len(risk_colors) - 1)]
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1, facecolor=color,
                                 edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            # Label
            label = str(risk_labels.get(val, str(val)))
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, label,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if val >= 2 else "black")

    # Axis labels
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(reversed(row_labels), fontsize=10)
    ax.set_xlabel("Consequence Severity →", fontsize=11)
    ax.set_ylabel("← Likelihood", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
        for c in risk_colors
    ]
    legend_labels = list(risk_labels.values())
    ax.legend(legend_patches, legend_labels, loc="upper right",
              bbox_to_anchor=(1.35, 1.02), fontsize=9, title="Risk Level")

    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="both", length=0)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Batch Export
# ══════════════════════════════════════════════════════════════════════════════

def export_all_plots(
    results: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    format: str = "png",
    dpi: int = 150,
) -> List[str]:
    """Export all plots from a project's results to an output directory.

    Parameters
    ----------
    results : list[dict]
        All scenario results, each potentially containing plot data.
    output_dir : str or Path
        Directory to save plots.
    format : str
        Image format.
    dpi : int
        Resolution.

    Returns
    -------
    list[str]
        Paths of all exported files.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for image export.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: List[str] = []

    for i, res in enumerate(results):
        prefix = f"{i+1:02d}_{res.get('type', 'plot')}"

        # Contour plot
        grid_data = res.get("grid_data", res)
        if all(k in grid_data for k in ("x", "y", "Z")):
            path = output_dir / f"{prefix}_contour.{format}"
            try:
                export_contour_plot(res, path, format=format, dpi=dpi,
                                    title=f"{res.get('name', 'Scenario')}")
                exported.append(str(path))
            except Exception:
                pass

        # FN curve
        fn_data = res.get("fn_data")
        if fn_data:
            path = output_dir / f"{prefix}_fn_curve.{format}"
            try:
                export_fn_curve(fn_data, path, format=format, dpi=dpi,
                                title=f"FN Curve — {res.get('name', 'Scenario')}")
                exported.append(str(path))
            except Exception:
                pass

        # Risk matrix
        risk_matrix = res.get("risk_matrix")
        if risk_matrix:
            path = output_dir / f"{prefix}_risk_matrix.{format}"
            try:
                export_risk_matrix_image(risk_matrix, path, format=format, dpi=dpi,
                                         title=f"Risk Matrix — {res.get('name', 'Scenario')}")
                exported.append(str(path))
            except Exception:
                pass

        # Pre-saved plot paths
        plots = res.get("plots", [])
        for j, plot in enumerate(plots):
            if isinstance(plot, dict):
                fig = plot.get("figure")
                if fig is not None:
                    path = output_dir / f"{prefix}_plot_{j+1}.{format}"
                    try:
                        fig.savefig(str(path), dpi=dpi, facecolor="white",
                                    edgecolor="none", bbox_inches="tight")
                        exported.append(str(path))
                    except Exception:
                        pass

    return exported
