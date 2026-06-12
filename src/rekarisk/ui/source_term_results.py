"""
Rekarisk UI — Source Term Results Panel.

Displays calculation results using tables and matplotlib plots.
Supports all source term result types (Orifice, Vessel, Pipe, PSV, Pool).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QSplitter, QScrollArea,
    QTextBrowser, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor

import numpy as np

# matplotlib for plotting (optional import at runtime)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Source Term Results Panel
# ══════════════════════════════════════════════════════════════════════════════

class SourceTermResultsPanel(QWidget):
    """Displays source term calculation results.

    Shows a summary table and, for time-series results (vessel, pool),
    a matplotlib plot of the key variables over time.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()
        self._current_result = None
        self._calc_type = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        self.header_label = QLabel("Calculation Results")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.header_label.setFont(font)
        layout.addWidget(self.header_label)

        # Summary table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        # Plot area (if matplotlib is available)
        if HAS_MPL:
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(250)
            layout.addWidget(self.canvas, 2)
        else:
            self.canvas = None

        # Messages
        self.messages_area = QTextBrowser()
        self.messages_area.setMaximumHeight(80)
        self.messages_area.setStyleSheet("QTextBrowser { color: #555555; }")
        layout.addWidget(self.messages_area)

        # Buttons
        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

    def show_orifice_result(self, result: Any):
        """Display orifice discharge results."""
        self._calc_type = "orifice"
        self.header_label.setText("🔩 Orifice Discharge Results")

        rows = [
            ("Mass flow rate", f"{result.mdot_initial:.4f} kg/s"),
            ("Phase", f"{result.phase}"),
            ("Exit velocity", f"{result.velocity:.2f} m/s"),
            ("Mass flux (G)", f"{result.G:.1f} kg/(m²·s)"),
            ("Is choked", f"{'Yes' if result.is_choked else 'No'}"),
            ("Flow regime", f"{result.flow_regime}"),
            ("P choked", f"{result.P_choked / 1e5:.3f} bar" if result.P_choked else "N/A"),
            ("P exit", f"{result.P_exit / 1e5:.3f} bar"),
            ("Orifice area", f"{result.area * 1e6:.1f} mm²"),
        ]
        if result.total_mass is not None:
            rows.append(("Total mass released", f"{result.total_mass:.2f} kg"))
        if result.flashing_fraction > 0:
            rows.append(("Flashing fraction", f"{result.flashing_fraction:.4f}"))

        self._fill_table(rows)
        self._clear_plot()
        self._fill_messages_str(getattr(result, "messages", ""))

    def show_vessel_result(self, result: Any):
        """Display vessel blowdown results."""
        self._calc_type = "vessel"
        self.header_label.setText("🛢️ Vessel Blowdown Results")

        rows = [
            ("Final pressure", f"{result.P[-1] / 1e5:.3f} bar"),
            ("Initial pressure", f"{result.P[0] / 1e5:.3f} bar"),
            ("Final temperature", f"{result.T[-1]:.1f} K"),
            ("Initial mass", f"{result.m[0]:.2f} kg"),
            ("Final mass", f"{result.m[-1]:.2f} kg"),
            ("Total released", f"{result.total_mass_released:.2f} kg"),
            ("Initial mdot", f"{result.mdot[0]:.4f} kg/s"),
            ("Final mdot", f"{result.mdot[-1]:.4f} kg/s"),
            ("Simulation time", f"{result.t_final:.1f} s"),
        ]
        if result.events:
            for key, val in result.events.items():
                rows.append((key, f"{val:.2f} s"))
        if result.phase_quality is not None:
            rows.append(("Initial quality", f"{result.phase_quality[0]:.4f}"))
            rows.append(("Final quality", f"{result.phase_quality[-1]:.4f}"))

        self._fill_table(rows)

        if HAS_MPL and len(result.t) > 1:
            self._plot_vessel(result)
        else:
            self._clear_plot()

        self._fill_messages_str(getattr(result, "messages", ""))

    def show_pipe_result(self, result: Any):
        """Display pipe flow results."""
        self._calc_type = "pipe"
        self.header_label.setText("🔧 Pipe Flow Results")

        rows = [
            ("Mass flow rate", f"{result.mdot:.4f} kg/s"),
            ("Exit velocity", f"{result.velocity:.2f} m/s"),
            ("Total ΔP", f"{result.delta_P / 1e5:.3f} bar"),
            ("Flow regime", f"{result.flow_regime}"),
            ("Friction factor", f"{result.friction_factor:.6f}"),
            ("Reynolds number", f"{result.Re:.0f}"),
            ("Is choked", f"{'Yes' if result.is_choked else 'No'}"),
        ]

        self._fill_table(rows)

        if HAS_MPL and len(result.P_profile) > 1:
            self._plot_pipe(result)
        else:
            self._clear_plot()

        self._fill_messages_str(getattr(result, "messages", ""))

    def show_psv_result(self, result: Any):
        """Display relief valve sizing results."""
        self._calc_type = "psv"
        self.header_label.setText("🔒 Relief Valve Sizing (API 520)")

        rows = [
            ("Required area", f"{result.A_required_mm2:.1f} mm²"),
            ("Orifice designation", result.orifice_designation),
            ("Relieving capacity", f"{result.W_relieving:.3f} kg/s"),
            ("Is choked", f"{'Yes' if result.is_choked else 'No'}"),
            ("Relieving pressure", f"{result.P_relieving / 1e5:.3f} bar abs"),
            ("Back pressure", f"{result.P_back_abs / 1e5:.3f} bar abs"),
        ]
        if hasattr(result, 'Kb'):
            rows.append(("Backpressure factor Kb", f"{result.Kb:.3f}"))

        self._fill_table(rows)
        self._clear_plot()
        self._fill_messages_str(getattr(result, "messages", ""))

    def show_pool_result(self, result: Any):
        """Display pool spreading and evaporation results."""
        self._calc_type = "pool"
        self.header_label.setText("💧 Pool Evaporation Results")

        rows = [
            ("Final radius", f"{result.pool_radius[-1]:.2f} m"),
            ("Final area", f"{result.pool_area[-1]:.1f} m²"),
            ("Final thickness", f"{result.pool_thickness[-1] * 1000:.1f} mm"),
            ("Avg evap rate", f"{result.avg_evap_rate:.6f} kg/(m²·s)"),
            ("Total evaporated", f"{result.total_evaporated:.2f} kg"),
            ("Mass remaining", f"{result.mass_remaining:.2f} kg"),
            ("Pool regime", result.pool_regime),
        ]

        self._fill_table(rows)

        if HAS_MPL and len(result.t) > 1:
            self._plot_pool(result)
        else:
            self._clear_plot()

        self._fill_messages_str(getattr(result, "messages", ""))

    def _fill_table(self, rows):
        self.table.setRowCount(len(rows))
        for i, (param, value) in enumerate(rows):
            param_item = QTableWidgetItem(str(param))
            param_item.setFont(QFont("", -1, QFont.Weight.Bold))
            self.table.setItem(i, 0, param_item)
            self.table.setItem(i, 1, QTableWidgetItem(str(value)))
        self.table.resizeRowsToContents()

    def _clear_plot(self):
        if self.canvas:
            self.figure.clear()
            self.canvas.draw()

    def _plot_vessel(self, result):
        """Plot vessel blowdown — P, T, mdot vs time."""
        self.figure.clear()
        t = result.t

        ax1 = self.figure.add_subplot(311)
        ax1.plot(t, result.P / 1e5, 'b-', linewidth=2)
        ax1.set_ylabel('Pressure [bar]')
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(312, sharex=ax1)
        ax2.plot(t, result.T, 'r-', linewidth=2)
        ax2.set_ylabel('Temperature [K]')
        ax2.grid(True, alpha=0.3)

        ax3 = self.figure.add_subplot(313, sharex=ax1)
        ax3.plot(t, result.mdot, 'g-', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('mdot [kg/s]')
        ax3.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_pipe(self, result):
        """Plot pipe pressure profile."""
        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.plot(result.x_profile, result.P_profile / 1e5, 'b-', linewidth=2)
        ax.set_xlabel('Distance along pipe [m]')
        ax.set_ylabel('Pressure [bar]')
        ax.set_title('Pressure Profile Along Pipe')
        ax.grid(True, alpha=0.3)
        ax.fill_between(result.x_profile, result.P_profile / 1e5, alpha=0.1, color='blue')

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_pool(self, result):
        """Plot pool behavior — radius, evap rate."""
        self.figure.clear()
        t = result.t

        ax1 = self.figure.add_subplot(211)
        ax1.plot(t, result.pool_radius, 'b-', linewidth=2)
        ax1.set_ylabel('Pool Radius [m]')
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.plot(t, result.evap_rate, 'r-', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Evap Rate [kg/s]')
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _fill_messages_str(self, messages):
        if not messages:
            self.messages_area.clear()
            return
        if isinstance(messages, list):
            text = "\n".join(messages)
        else:
            text = str(messages)
        self.messages_area.setText(text)

    def clear(self):
        """Clear all results."""
        self._calc_type = None
        self._current_result = None
        self.header_label.setText("Calculation Results")
        self.table.setRowCount(0)
        self._clear_plot()
        self.messages_area.clear()
        self.export_btn.setEnabled(False)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone Results Display (for embedding in dock/widget)
# ══════════════════════════════════════════════════════════════════════════════

class SourceTermResultsDock(QWidget):
    """Widget combining results panel with export/print capability."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.results = SourceTermResultsPanel()
        layout.addWidget(self.results)
