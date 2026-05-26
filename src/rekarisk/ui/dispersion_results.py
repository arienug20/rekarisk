"""
Rekarisk UI — Dispersion Results Panel.

Displays dispersion calculation results using tables and matplotlib plots:
  - 2D ground-level concentration contour (bird's eye view)
  - Centerline concentration profile (C vs downwind distance)
  - Cross-wind concentration profile (C vs cross-wind distance)
  - Concentration vs time (for puff models)
  - Summary table with key metrics
  - CSV export functionality

Uses PyQt5 for the UI framework and matplotlib for plotting.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QSplitter, QScrollArea,
    QTextBrowser, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor

# matplotlib for plotting
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# MPL Setup
# ══════════════════════════════════════════════════════════════════════════════

if HAS_MPL:
    # Use Qt5Agg backend
    import matplotlib
    matplotlib.use('Qt5Agg')


# ══════════════════════════════════════════════════════════════════════════════
# Dispersion Results Panel
# ══════════════════════════════════════════════════════════════════════════════

class DispersionResultsPanel(QWidget):
    """Displays dispersion calculation results with plots and tables.

    Tabs:
      - Contour: 2D ground-level concentration contour map
      - Centerline: Centerline concentration vs downwind distance
      - Cross-wind: Cross-wind profile at selected distance
      - Time: Concentration vs time (puff model only)
      - Summary: Table of key metrics
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_result = None
        self._model_type = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QHBoxLayout()
        self.header_label = QLabel("Dispersion Results")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.header_label.setFont(font)
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        # Export button
        self.export_btn = QPushButton("📥 Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)

        layout.addLayout(header_layout)

        # Model info banner
        self.model_label = QLabel("")
        self.model_label.setStyleSheet(
            "background-color: #E3F2FD; padding: 6px; border-radius: 4px;"
        )
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        # Tab widget for results
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create result views
        self._contour_tab = ContourPlotTab()
        self._centerline_tab = CenterlinePlotTab()
        self._crosswind_tab = CrosswindPlotTab()
        self._time_tab = TimeSeriesPlotTab()
        self._summary_tab = SummaryTab()

        self.tabs.addTab(self._contour_tab, "🗺️ Contour")
        self.tabs.addTab(self._centerline_tab, "📈 Centerline")
        self.tabs.addTab(self._crosswind_tab, "↔️ Cross-wind")
        self.tabs.addTab(self._time_tab, "⏱️ Time Series")
        self.tabs.addTab(self._summary_tab, "📋 Summary")

        layout.addWidget(self.tabs)

    def display_result(
        self,
        result: Any,
        model_type: str = "unknown",
        params: Optional[Dict[str, Any]] = None,
    ):
        """Display a dispersion calculation result.

        Args:
            result: Any dispersion result object (PlumeResult, PuffResult,
                    DenseGasResult, or DispatchResult).
            model_type: Type of model used (e.g., 'gaussian_plume').
            params: Optional input parameters for context.
        """
        self._current_result = result
        self._model_type = model_type
        self.export_btn.setEnabled(True)

        # Check for DispatchResult wrapper
        dispatch_result = None
        plume_result = None
        puff_result = None
        dense_result = None

        from rekarisk.models.dispersion.dispersion_dispatcher import DispatchResult
        from rekarisk.models.dispersion.gaussian_plume import PlumeResult
        from rekarisk.models.dispersion.gaussian_puff import PuffResult
        from rekarisk.models.dispersion.dense_gas import DenseGasResult

        if isinstance(result, DispatchResult):
            dispatch_result = result
            plume_result = result.plume_result
            puff_result = result.puff_result
            dense_result = result.dense_result
            self.model_label.setText(result.message)
        elif isinstance(result, PlumeResult):
            plume_result = result
            self.model_label.setText(
                f"Gaussian Plume | Max: {result.max_concentration:.2f} mg/m³ "
                f"at {result.max_distance:.0f} m"
            )
        elif isinstance(result, PuffResult):
            puff_result = result
            self.model_label.setText(
                f"Gaussian Puff | {len(result.time_series)} time steps "
                f"| Peak: {np.max(result.max_concentration_over_time):.2f} mg/m³"
            )
        elif isinstance(result, DenseGasResult):
            dense_result = result
            self.model_label.setText(
                f"Dense Gas | Max: {result.max_concentration:.2f} mg/m³ "
                f"| Transition at {result.transition_distance:.0f} m "
                f"({result.transition_time:.0f} s)"
            )
        elif isinstance(result, dict):
            self.model_label.setText(f"Raw Results: {model_type}")

        # Display in appropriate tabs
        if plume_result is not None:
            self._contour_tab.display_plume(plume_result)
            self._centerline_tab.display_plume(plume_result)
            self._crosswind_tab.display_plume(plume_result)
            self._time_tab.clear()
            self._summary_tab.display_plume(plume_result)

        if puff_result is not None:
            self._time_tab.display_puff(puff_result)
            self._contour_tab.clear()
            self._centerline_tab.clear()
            self._crosswind_tab.clear()
            self._summary_tab.display_puff(puff_result)

        if dense_result is not None and plume_result is None and puff_result is None:
            self._contour_tab.display_dense_gas(dense_result)
            self._centerline_tab.clear()
            self._crosswind_tab.clear()
            self._time_tab.display_dense_gas(dense_result)
            self._summary_tab.display_dense_gas(dense_result)

    def display_error(self, message: str):
        """Display an error message."""
        self.model_label.setText(f"❌ Error: {message}")
        self.model_label.setStyleSheet(
            "background-color: #FFEBEE; padding: 6px; border-radius: 4px;"
        )
        self._contour_tab.clear()
        self._centerline_tab.clear()
        self._crosswind_tab.clear()
        self._time_tab.clear()
        self._summary_tab.clear()
        self.export_btn.setEnabled(False)

    def _export_csv(self):
        """Export results to CSV."""
        if self._current_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Dispersion Results",
            "dispersion_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            self._write_csv(path)
            QMessageBox.information(
                self, "Export Complete",
                f"Results exported to:\n{path}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Export Error",
                f"Failed to export: {str(e)}"
            )

    def _write_csv(self, path: str):
        """Write results to CSV file."""
        result = self._current_result

        from rekarisk.models.dispersion.gaussian_plume import PlumeResult

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            if isinstance(result, PlumeResult):
                # Export ground concentration grid
                writer.writerow(["# Gaussian Plume Results"])
                writer.writerow(["# Max Concentration (mg/m3)", f"{result.max_concentration:.4f}"])
                writer.writerow(["# Max Distance (m)", f"{result.max_distance:.2f}"])
                writer.writerow(["# Plume Rise (m)", f"{result.plume_rise_delta:.2f}"])
                writer.writerow([])
                writer.writerow(["# Ground Concentration Grid (y vs x)"])
                writer.writerow(["x\\y"] + [f"{y:.1f}" for y in result.y_coords])

                for i, x in enumerate(result.x_coords):
                    row = [f"{x:.1f}"] + [f"{v:.6f}" for v in result.ground_concentration[i, :]]
                    writer.writerow(row)

                writer.writerow([])
                writer.writerow(["# Centerline Profile"])
                writer.writerow(["Distance (m)", "Concentration (mg/m3)"])
                for x, c in zip(result.x_coords, result.centerline_concentration):
                    writer.writerow([f"{x:.1f}", f"{c:.6f}"])

            elif hasattr(result, 'to_dict'):
                d = result.to_dict()
                writer.writerow(["# Dispersion Results"])
                for key, value in d.items():
                    if isinstance(value, list):
                        writer.writerow([f"# {key}", str(value)])
                    else:
                        writer.writerow([f"# {key}", str(value)])
            else:
                writer.writerow(["# Dispersion Results"])
                writer.writerow(["# Result type:", type(result).__name__])
                writer.writerow([str(result)])


# ══════════════════════════════════════════════════════════════════════════════
# Contour Plot Tab (Bird's Eye View)
# ══════════════════════════════════════════════════════════════════════════════

class ContourPlotTab(QWidget):
    """2D ground-level concentration contour plot."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib is required for plots."))
            return

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def display_plume(self, result) -> None:
        """Display ground concentration contour for Gaussian plume."""
        if not HAS_MPL:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ground_C = result.ground_concentration
        x = result.x_coords
        y = result.y_coords

        X, Y = np.meshgrid(x, y, indexing='ij')

        # Log-spaced contours
        if ground_C.max() > 0:
            levels = np.logspace(
                max(np.log10(ground_C[ground_C > 0].min()), -3),
                np.log10(ground_C.max()),
                15
            )
        else:
            levels = 10

        contour = ax.contourf(X, Y, ground_C, levels=levels, cmap='hot')
        cbar = self.figure.colorbar(contour, ax=ax, label='Concentration [mg/m³]')
        ax.contour(X, Y, ground_C, levels=levels[::3], colors='cyan',
                   linewidths=0.5, alpha=0.5)

        ax.set_xlabel('Downwind Distance [m]')
        ax.set_ylabel('Cross-wind Distance [m]')
        ax.set_title('Ground-Level Concentration (z=0)')
        ax.grid(True, alpha=0.3)

        # Mark max concentration point
        max_idx = np.unravel_index(np.argmax(ground_C), ground_C.shape)
        ax.plot(x[max_idx[0]], y[max_idx[1]], 'bx', markersize=10, markeredgewidth=2,
                label=f'Max: {ground_C[max_idx]:.2f} mg/m³ at {x[max_idx[0]]:.0f} m')
        ax.legend(loc='upper right')

        self.canvas.draw()

    def display_dense_gas(self, result) -> None:
        """Display dense gas results."""
        if not HAS_MPL:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        times = np.array([r.time for r in result.time_series])
        radii = np.array([r.radius for r in result.time_series])
        heights = np.array([r.height for r in result.time_series])

        ax.plot(times, radii, 'b-', label='Cloud Radius', linewidth=2)
        ax.plot(times, heights, 'r--', label='Cloud Height', linewidth=2)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Dimension [m]')
        ax.set_title('Dense Gas Cloud Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Transition marker
        if result.transition_time > 0:
            ax.axvline(x=result.transition_time, color='green', linestyle=':',
                       alpha=0.7, label=f'Transition ({result.transition_time:.0f}s)')
            ax.legend()

        self.canvas.draw()

    def clear(self):
        if HAS_MPL:
            self.figure.clear()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Centerline Profile Tab
# ══════════════════════════════════════════════════════════════════════════════

class CenterlinePlotTab(QWidget):
    """Centerline concentration vs downwind distance plot."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib is required for plots."))
            return

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def display_plume(self, result) -> None:
        """Display centerline concentration profile."""
        if not HAS_MPL:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x = result.x_coords
        cl = result.centerline_concentration

        ax.semilogy(x, cl, 'b-', linewidth=2)
        ax.fill_between(x, cl, alpha=0.15, color='blue')

        ax.set_xlabel('Downwind Distance [m]')
        ax.set_ylabel('Concentration [mg/m³]')
        ax.set_title('Centerline Concentration Profile (y=0, z=0)')
        ax.grid(True, alpha=0.3, which='both')

        # Mark max
        max_idx = np.argmax(cl)
        ax.plot(x[max_idx], cl[max_idx], 'ro', markersize=8,
                label=f'Max: {cl[max_idx]:.2f} mg/m³ at {x[max_idx]:.0f} m')
        ax.legend()

        self.canvas.draw()

    def clear(self):
        if HAS_MPL:
            self.figure.clear()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Cross-wind Profile Tab
# ══════════════════════════════════════════════════════════════════════════════

class CrosswindPlotTab(QWidget):
    """Cross-wind concentration profile at selected downwind distance."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib is required for plots."))
            return

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def display_plume(self, result) -> None:
        """Display cross-wind profile at multiple downwind distances."""
        if not HAS_MPL:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ground_C = result.ground_concentration
        x = result.x_coords
        y = result.y_coords

        # Plot cross-wind profiles at several distances
        n_x = len(x)
        if n_x > 5:
            indices = np.linspace(0, n_x - 1, min(6, n_x), dtype=int)
        else:
            indices = list(range(n_x))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, ci in enumerate(indices):
            color = colors[idx % len(colors)]
            ax.plot(y, ground_C[ci, :], '-', color=color, linewidth=2,
                    label=f'x = {x[ci]:.0f} m')

        ax.set_xlabel('Cross-wind Distance [m]')
        ax.set_ylabel('Concentration [mg/m³]')
        ax.set_title('Cross-wind Concentration Profiles (z=0)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        self.canvas.draw()

    def clear(self):
        if HAS_MPL:
            self.figure.clear()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Time Series Tab (for Puff & Dense Gas)
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesPlotTab(QWidget):
    """Concentration vs time plot for puff and dense gas models."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib is required for plots."))
            return

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def display_puff(self, result) -> None:
        """Display puff concentration over time."""
        if not HAS_MPL:
            return

        self.figure.clear()

        # Max concentration over time
        ax1 = self.figure.add_subplot(211)
        ax1.plot(result.times, result.max_concentration_over_time, 'b-', linewidth=2)
        ax1.set_ylabel('Max Concentration [mg/m³]')
        ax1.set_title('Puff — Peak Concentration vs Time')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(result.times, result.max_concentration_over_time,
                          alpha=0.15, color='blue')

        # Puff center position
        ax2 = self.figure.add_subplot(212)
        ax2.plot(result.times, result.puff_center_positions[:, 0],
                 'r-', label='Center X', linewidth=2)
        ax2.plot(result.times, result.puff_center_positions[:, 1],
                 'g--', label='Center Y', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Position [m]')
        ax2.set_title('Puff Center Position')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def display_dense_gas(self, result) -> None:
        """Display dense gas concentration and density over time."""
        if not HAS_MPL:
            return

        self.figure.clear()

        # Concentration over time
        ax1 = self.figure.add_subplot(211)
        times = np.array([r.time for r in result.time_series])
        conc = np.array([r.concentration_center for r in result.time_series])
        ax1.semilogy(times, conc, 'b-', linewidth=2)
        ax1.set_ylabel('Concentration [mg/m³]')
        ax1.set_title('Dense Gas — Centerline Concentration vs Time')
        ax1.grid(True, alpha=0.3, which='both')

        # Density ratio over time
        ax2 = self.figure.add_subplot(212)
        dr = np.array([r.density_ratio for r in result.time_series])
        ax2.plot(times, dr, 'r-', linewidth=2)
        ax2.axhline(y=1.01, color='gray', linestyle=':', alpha=0.7,
                    label='Passive threshold (1.01)')
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
                    label='ρ_c = ρ_a')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('ρ_c / ρ_a')
        ax2.set_title('Density Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        if HAS_MPL:
            self.figure.clear()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Summary Tab
# ══════════════════════════════════════════════════════════════════════════════

class SummaryTab(QWidget):
    """Summary table of key dispersion metrics."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.table)

    def display_plume(self, result) -> None:
        """Display Gaussian plume summary."""
        metrics = [
            ("Model", "Gaussian Plume (Continuous)"),
            ("Source Rate", f"{result.input.source_rate if result.input else 'N/A'} kg/s"),
            ("Wind Speed", f"{result.input.wind_speed if result.input else 'N/A'} m/s"),
            ("Stability Class", f"{result.input.stability_class if result.input else 'N/A'}"),
            ("Release Height", f"{result.input.release_height if result.input else 'N/A'} m"),
            ("Plume Rise", f"{result.plume_rise_delta:.1f} m"),
            ("", ""),
            ("Max Concentration", f"{result.max_concentration:.4f} mg/m³"),
            ("Max Distance", f"{result.max_distance:.1f} m"),
            ("Max Y Offset", f"{result.max_y_distance:.1f} m"),
            ("Max Z Height", f"{result.max_z_height:.1f} m"),
            ("", ""),
            ("Grid Shape", f"{result.grid_shape[0]} × {result.grid_shape[1]} × {result.grid_shape[2]}"),
            ("X Range", f"{result.x_coords[0]:.0f} – {result.x_coords[-1]:.0f} m"),
            ("Y Range", f"{result.y_coords[0]:.0f} – {result.y_coords[-1]:.0f} m"),
            ("Z Range", f"{result.z_coords[0]:.0f} – {result.z_coords[-1]:.0f} m"),
        ]
        self._populate_table(metrics)

    def display_puff(self, result) -> None:
        """Display Gaussian puff summary."""
        n_steps = len(result.time_series)
        max_C = float(np.max(result.max_concentration_over_time))
        max_t_idx = int(np.argmax(result.max_concentration_over_time))
        max_t = result.times[max_t_idx] if max_t_idx < len(result.times) else 0
        total_dose_max = float(np.max(result.total_dose))

        metrics = [
            ("Model", "Gaussian Puff"),
            ("Total Mass", f"{result.input.mass if result.input else 'N/A'} kg"),
            ("Duration", f"{result.input.release_duration if result.input else 'N/A'} s"),
            ("Wind Speed", f"{result.input.wind_speed if result.input else 'N/A'} m/s"),
            ("Wind Dir", f"{result.input.wind_direction if result.input else 'N/A'}°"),
            ("Stability", f"{result.input.stability_class if result.input else 'N/A'}"),
            ("", ""),
            ("Time Steps", str(n_steps)),
            ("Peak Concentration", f"{max_C:.4f} mg/m³"),
            ("Peak Time", f"{max_t:.1f} s"),
            ("Total Dose (max)", f"{total_dose_max:.4f} mg·s/m³"),
        ]
        self._populate_table(metrics)

    def display_dense_gas(self, result) -> None:
        """Display dense gas summary."""
        metrics = [
            ("Model", "Dense Gas (SLAB-based)"),
            ("Release Type", f"{result.input.release_type if result.input else 'N/A'}"),
            ("Total Mass", f"{result.input.total_mass if result.input else 'N/A':.1f} kg"),
            ("Initial ρ_c/ρ_a", f"{result.initial_density_ratio:.3f}"),
            ("", ""),
            ("Max Concentration", f"{result.max_concentration:.2f} mg/m³"),
            ("Transition Distance", f"{result.transition_distance:.1f} m"),
            ("Transition Time", f"{result.transition_time:.1f} s"),
            ("Cloud Radius at Transition", f"{result.transition_radius:.1f} m"),
            ("Cloud Height at Transition", f"{result.transition_height:.1f} m"),
            ("ρ_c/ρ_a at Transition", f"{result.transition_density_ratio:.3f}"),
            ("", ""),
            ("Simulation Steps", str(len(result.time_series))),
            ("Final Density Ratio",
             f"{result.density_ratios[-1]:.3f}" if len(result.density_ratios) > 0
             else "N/A"),
        ]
        self._populate_table(metrics)

    def clear(self):
        self.table.setRowCount(0)

    def _populate_table(self, metrics: List[Tuple[str, str]]):
        """Populate table with metric key-value pairs."""
        self.table.setRowCount(len(metrics))
        for i, (key, value) in enumerate(metrics):
            key_item = QTableWidgetItem(key)
            if key == "":
                key_item = QTableWidgetItem("")
            else:
                font = key_item.font()
                font.setBold(True)
                key_item.setFont(font)
            self.table.setItem(i, 0, key_item)
            self.table.setItem(i, 1, QTableWidgetItem(value))
