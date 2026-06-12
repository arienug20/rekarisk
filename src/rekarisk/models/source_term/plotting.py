"""
Rekarisk — Source Term Plotting Utilities.

Provides visualization functions for source term simulation results:
vessel blowdown (P/T/m vs time), orifice discharge, orifice comparison,
and multi-scenario overlays.

Uses matplotlib with a clean, publication-ready style.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Unit conversions (self-contained to avoid circular imports)
PSI2PA = 6894.757293168
K2F = lambda k: (k - 273.15) * 9 / 5 + 32
K2C = lambda k: k - 273.15
LB2KG = 0.45359237

# Style defaults
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
}


def _ensure_dir(path: str) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_blowdown_profile(
    t: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    m: np.ndarray,
    mdot: np.ndarray,
    *,
    title: str = "Vessel Blowdown Profile",
    subtitle: str = "",
    save_path: Optional[str] = None,
    format: str = "png",
    dpi: int = 150,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Generate a 4-panel blowdown profile plot.

    Panels:
      (1) Pressure vs Time [psia]
      (2) Temperature vs Time [°F]
      (3) Mass Remaining vs Time [lb]
      (4) Mass Flow Rate vs Time [lb/hr]

    Args:
        t: Time array [s].
        P: Pressure array [Pa].
        T: Temperature array [K].
        m: Mass array [kg].
        mdot: Mass flow rate array [kg/s].
        title: Main title.
        subtitle: Subtitle (e.g. vessel name, parameters).
        save_path: File path to save (auto-generates if None and outputs/ exists).
        format: Image format (png, pdf, svg).
        dpi: Resolution for raster formats.
        show: Whether to display interactively (often False in headless).

    Returns:
        matplotlib Figure, or None if save_path is None and no outputs/ dir.
    """
    if save_path is None:
        try:
            outdir = _ensure_dir(
                os.path.join(os.getcwd(), "..", "..", "..", "outputs")
            )
            save_path = str(outdir / f"blowdown_profile.{format}")
        except Exception:
            pass

    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=9, color="gray")

    # Panel 1: Pressure
    ax = axes[0, 0]
    ax.plot(t / 60, P / PSI2PA, color="#e74c3c", linewidth=2)
    ax.set_ylabel("Pressure [psia]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Pressure vs Time")
    ax.fill_between(t / 60, P / PSI2PA, alpha=0.15, color="#e74c3c")

    # Panel 2: Temperature
    ax = axes[0, 1]
    T_f = (T - 273.15) * 9 / 5 + 32
    ax.plot(t / 60, T_f, color="#2980b9", linewidth=2)
    ax.set_ylabel("Temperature [°F]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Temperature vs Time")
    ax.fill_between(t / 60, T_f, alpha=0.15, color="#2980b9")

    # Panel 3: Mass remaining
    ax = axes[1, 0]
    m_lb = m * 2.20462
    ax.plot(t / 60, m_lb, color="#27ae60", linewidth=2)
    ax.set_ylabel("Mass Remaining [lb]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Mass vs Time")
    ax.fill_between(t / 60, m_lb, alpha=0.15, color="#27ae60")
    # Add initial and final annotations
    ax.annotate(f"{m_lb[0]:.1f} lb", xy=(t[0] / 60, m_lb[0]),
                xytext=(5, 10), textcoords="offset points", fontsize=8)
    ax.annotate(f"{m_lb[-1]:.1f} lb", xy=(t[-1] / 60, m_lb[-1]),
                xytext=(-70, -15), textcoords="offset points", fontsize=8)

    # Panel 4: Mass flow rate
    ax = axes[1, 1]
    mdot_lb_hr = mdot * 3600 * 2.20462
    ax.plot(t / 60, mdot_lb_hr, color="#8e44ad", linewidth=2)
    ax.set_ylabel("Mass Flow Rate [lb/hr]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Mass Flow Rate vs Time")
    ax.fill_between(t / 60, mdot_lb_hr, alpha=0.15, color="#8e44ad")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=format)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_comparison(
    scenarios: dict[str, dict],
    *,
    title: str = "Blowdown Scenario Comparison",
    save_path: Optional[str] = None,
    format: str = "png",
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """Overlay multiple blowdown scenarios on single axes.

    Args:
        scenarios: Dict of label → dict with keys t, P, T, m, mdot (all arrays).
        title: Figure title.
        save_path: File path to save.
        format: Image format.
        dpi: Resolution.

    Returns:
        matplotlib Figure.
    """
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#1abc9c"]
    color_cycle = iter(colors * 3)

    # Panel 1: Pressure
    ax = axes[0, 0]
    for label, data in scenarios.items():
        c = next(color_cycle)
        ax.plot(data["t"] / 60, data["P"] / PSI2PA, color=c, label=label, linewidth=2)
    ax.set_ylabel("Pressure [psia]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Pressure Comparison")
    ax.legend()

    # Panel 2: Temperature
    ax = axes[0, 1]
    color_cycle = iter(colors * 3)
    for label, data in scenarios.items():
        c = next(color_cycle)
        T_f = K2F(data["T"])
        ax.plot(data["t"] / 60, T_f, color=c, label=label, linewidth=2)
    ax.set_ylabel("Temperature [°F]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Temperature Comparison")
    ax.legend()

    # Panel 3: Mass
    ax = axes[1, 0]
    color_cycle = iter(colors * 3)
    for label, data in scenarios.items():
        c = next(color_cycle)
        m_lb = data["m"] * 2.20462
        ax.plot(data["t"] / 60, m_lb, color=c, label=label, linewidth=2)
    ax.set_ylabel("Mass [lb]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Mass Comparison")
    ax.legend()

    # Panel 4: Mass flow rate
    ax = axes[1, 1]
    color_cycle = iter(colors * 3)
    for label, data in scenarios.items():
        c = next(color_cycle)
        mdot_lb_hr = data["mdot"] * 3600 * 2.20462
        ax.plot(data["t"] / 60, mdot_lb_hr, color=c, label=label, linewidth=2)
    ax.set_ylabel("Mass Flow Rate [lb/hr]")
    ax.set_xlabel("Time [min]")
    ax.set_title("Flow Rate Comparison")
    ax.legend()

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=format)
        print(f"  📊 Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_blowdown_summary(
    t: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    m: np.ndarray,
    mdot: np.ndarray,
    *,
    Z: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    T_wall: Optional[np.ndarray] = None,
    vessel_name: str = "",
    orifice_mm: float = 0.0,
    save_path: Optional[str] = None,
    format: str = "png",
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """Six-panel comprehensive blowdown summary with Z/k diagnostics.

    Args:
        t, P, T, m, mdot: Standard blowdown arrays.
        Z: Optional Z-factor history (for dynamic EOS runs).
        k: Optional k (Cp/Cv) history (for dynamic EOS runs).
        vessel_name: Vessel label.
        orifice_mm: Orifice diameter in mm (for title).
        save_path: Save destination.
        format: Image format.
        dpi: Resolution.

    Returns:
        matplotlib Figure.
    """
    has_diag = Z is not None and k is not None
    nrows = 2 if not has_diag else 3
    ncols = 3

    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows + 0.5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else np.array(axes).flatten()

    title_str = f"Blowdown — {vessel_name}"
    if orifice_mm > 0:
        title_str += f" ({orifice_mm} mm orifice)"
    fig.suptitle(title_str, fontsize=14, fontweight="bold")

    t_sec = t  # time in seconds
    t_plot = t_sec  # plot in seconds
    t_label = "Time [s]"

    # (1) Pressure
    ax = axes[0]
    ax.plot(t_plot, P / PSI2PA, color="#e74c3c", linewidth=2)
    ax.fill_between(t_plot, P / PSI2PA, alpha=0.12, color="#e74c3c")
    ax.set_ylabel("P [psia]"); ax.set_xlabel(t_label)
    ax.set_title("Pressure")

    # (2) Temperature
    ax = axes[1]
    T_f = K2F(T)
    ax.plot(t_plot, T_f, color="#2980b9", linewidth=2, label="Gas")
    ax.fill_between(t_plot, T_f, alpha=0.12, color="#2980b9")
    if T_wall is not None:
        Tw_f = K2F(T_wall)
        ax.plot(t_plot, Tw_f, color="#e67e22", linewidth=2, linestyle="--", label="Wall")
        ax.fill_between(t_plot, Tw_f, alpha=0.08, color="#e67e22")
        ax.legend(fontsize=7)
    ax.set_ylabel("T [°F]"); ax.set_xlabel(t_label)
    ax.set_title("Temperature")

    # (3) Mass
    ax = axes[2]
    m_lb = m * 2.20462
    ax.plot(t_plot, m_lb, color="#27ae60", linewidth=2)
    ax.fill_between(t_plot, m_lb, alpha=0.12, color="#27ae60")
    ax.set_ylabel("m [lb]"); ax.set_xlabel(t_label)
    ax.set_title("Mass Remaining")

    # (4) Mass flow rate
    ax = axes[3]
    mdot_lb_hr = mdot * 3600 * 2.20462
    ax.plot(t_plot, mdot_lb_hr, color="#8e44ad", linewidth=2)
    ax.fill_between(t_plot, mdot_lb_hr, alpha=0.12, color="#8e44ad")
    ax.set_ylabel("ṁ [lb/hr]"); ax.set_xlabel(t_label)
    ax.set_title("Mass Flow Rate")

    # (5) Phase envelope (P-T diagram)
    ax = axes[4]
    ax.plot(T_f, P / PSI2PA, color="#e67e22", linewidth=2)
    ax.scatter([T_f[0]], [P[0] / PSI2PA], color="#e74c3c", s=60, zorder=5, label="Start")
    ax.scatter([T_f[-1]], [P[-1] / PSI2PA], color="#2980b9", s=60, zorder=5, label="End")
    ax.set_xlabel("T [°F]"); ax.set_ylabel("P [psia]")
    ax.set_title("P-T Trajectory")
    ax.legend(fontsize=7)

    # (6) Diagnostics or additional info
    ax = axes[5]
    if has_diag:
        ax2 = ax.twinx()
        l1 = ax.plot(t_plot, Z, color="#2c3e50", linewidth=2, label="Z")
        l2 = ax2.plot(t_plot, k, color="#f39c12", linewidth=2, label="k")
        ax.set_ylabel("Z [-]", color="#2c3e50")
        ax2.set_ylabel("k (Cp/Cv) [-]", color="#f39c12")
        ax.set_xlabel(t_label)
        ax.set_title("Dynamic Z & k")
        ax.tick_params(axis="y", labelcolor="#2c3e50")
        ax2.tick_params(axis="y", labelcolor="#f39c12")
        lines = l1 + l2
        labels = [ll.get_label() for ll in lines]
        ax.legend(lines, labels, fontsize=7)
    else:
        # Summary stats table
        dm = (m[0] - m[-1]) * 2.20462
        dt_blow = t[-1] / 60
        T_drop = T[0] - T[-1]
        T_drop_F = K2F(T[0]) - K2F(T[-1])
        stats_text = (
            f"Summary:\n"
            f"  Blowdown: {dt_blow*60:.0f} s\n"
            f"  Δm: {dm:.1f} lb\n"
            f"  ΔT_gas: {T_drop_F:.0f} °F\n"
            f"  ṁ_max: {np.max(mdot)*3600*2.20462:.0f} lb/hr\n"
            f"  P_f: {P[-1]/PSI2PA:.0f} psia"
        )
        if T_wall is not None:
            Tw_f = K2F(T_wall)
            stats_text += f"\n  T_gas_min: {np.min(T_f):.0f}°F\n  T_wall_min: {np.min(Tw_f):.0f}°F"
        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
                fontsize=10, fontfamily="monospace",
                verticalalignment="center", horizontalalignment="center",
                bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Statistics")

    if has_diag and len(axes) > 6:
        # Hide extra axes
        for ax in axes[6:]:
            ax.set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=format)
        print(f"  📊 Saved: {save_path}")

    plt.close(fig)
    return fig
