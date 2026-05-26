"""
Rekarisk UI — Explosion Results Panel.

Displays explosion consequence analysis results using tables and plots:
  - Overpressure vs distance (log-log) with all methods overlaid
  - Threshold lines (1, 3, 5, 8, 10 psi)
  - Distance-to-threshold table
  - Summary metrics table
  - CSV export
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QSplitter,
    QTextBrowser, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    import matplotlib
    matplotlib.use("QtAgg")


PSI_TO_KPA = 6.89475729
DAMAGE_THRESHOLDS_PSI = [1.0, 3.0, 5.0, 8.0, 10.0]
THRESHOLD_COLORS = {
    1.0: "#4CAF50",
    3.0: "#FFC107",
    5.0: "#FF9800",
    8.0: "#F44336",
    10.0: "#B71C1C",
}
THRESHOLD_LABELS = {
    1.0: "1 psi — Minor glass breakage",
    3.0: "3 psi — 50% window breakage",
    5.0: "5 psi — 95% window breakage",
    8.0: "8 psi — Minor structural damage",
    10.0: "10 psi — Steel frame distortion",
}

MODEL_COLORS = {
    "TNT": "#2196F3",
    "TNO": "#FF9800",
    "BST": "#4CAF50",
}


# ══════════════════════════════════════════════════════════════════════════════
# Explosion Results Panel
# ══════════════════════════════════════════════════════════════════════════════

class ExplosionResultsPanel(QWidget):
    """Displays explosion results with plots and tables.

    Tabs:
      - Overpressure Plot: P_s vs R (log-log) with threshold lines
      - Thresholds Table: Distance to each damage threshold
      - Summary: Key metrics per model
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._results: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QHBoxLayout()
        self.header_label = QLabel("Explosion Analysis Results")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.header_label.setFont(font)
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        self.export_btn = QPushButton("📥 Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)

        layout.addLayout(header_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._plot_tab = OverpressurePlotTab()
        self._threshold_tab = ThresholdTableTab()
        self._summary_tab = ExplosionSummaryTab()

        self.tabs.addTab(self._plot_tab, "📈 Overpressure vs Distance")
        self.tabs.addTab(self._threshold_tab, "🎯 Damage Thresholds")
        self.tabs.addTab(self._summary_tab, "📋 Summary")

        layout.addWidget(self.tabs)

    def display_results(
        self,
        results: Dict[str, Any],
    ):
        """Display explosion calculation results.

        Args:
            results: Dict with keys like 'tnt', 'tno', 'bst', each
                     containing an ExplosionResult or None.
        """
        self._results = results
        self.export_btn.setEnabled(True)

        # Update plot
        self._plot_tab.display(results)

        # Update threshold table
        self._threshold_tab.display(results)

        # Update summary
        self._summary_tab.display(results)

    def display_error(self, message: str):
        """Display an error message."""
        self.header_label.setText("❌ Error")
        self._plot_tab.clear()
        self._threshold_tab.clear()
        self._summary_tab.clear()
        self.export_btn.setEnabled(False)

    def _export_csv(self):
        """Export all results to CSV."""
        if not self._results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Explosion Results",
            "explosion_results.csv",
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
        """Write results to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            for model_key, result in self._results.items():
                if result is None:
                    continue

                writer.writerow([f"# {result.model_name}"])
                writer.writerow([
                    "# Distance (m)", "Overpressure (kPa)",
                    "Impulse (kPa·ms)", "Duration (ms)"
                ])

                for i in range(len(result.distances)):
                    writer.writerow([
                        f"{result.distances[i]:.2f}",
                        f"{result.overpressure[i]:.4f}",
                        f"{result.impulse[i]:.4f}",
                        f"{result.positive_phase_duration[i]:.4f}",
                    ])

                writer.writerow([])
                writer.writerow(["# Damage Thresholds"])
                writer.writerow(["Threshold (psi)", "Distance (m)"])
                for psi, dist in sorted(result.distance_to_thresholds.items()):
                    writer.writerow([f"{psi}", f"{dist:.2f}"])
                writer.writerow([])


# ══════════════════════════════════════════════════════════════════════════════
# Overpressure Plot Tab
# ══════════════════════════════════════════════════════════════════════════════

class OverpressurePlotTab(QWidget):
    """Overpressure vs distance plot (log-log) with threshold lines."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib is required for plots."))
            return

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def display(self, results: Dict[str, Any]):
        """Plot overpressure vs distance for all models."""
        if not HAS_MPL:
            return

        self.figure.clear()

        # Main overpressure plot
        ax = self.figure.add_subplot(111)

        has_data = False

        for model_key, result in results.items():
            if result is None:
                continue
            has_data = True

            color = MODEL_COLORS.get(model_key.upper(), "#757575")
            label = result.model_name

            ax.loglog(
                result.distances, result.overpressure,
                "-", color=color, linewidth=2, label=label, alpha=0.85
            )

        if not has_data:
            self.canvas.draw()
            return

        # Damage threshold lines
        for psi in DAMAGE_THRESHOLDS_PSI:
            kpa = psi * PSI_TO_KPA
            color = THRESHOLD_COLORS.get(psi, "#999")
            ax.axhline(
                y=kpa, color=color, linestyle="--",
                linewidth=1, alpha=0.6,
                label=THRESHOLD_LABELS.get(psi, f"{psi} psi")
            )

        ax.set_xlabel("Distance [m]", fontsize=11)
        ax.set_ylabel("Peak Side-On Overpressure [kPa]", fontsize=11)
        ax.set_title("Overpressure vs Distance — Explosion Blast Analysis", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Add secondary y-axis in psi
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        y_min, y_max = ax.get_ylim()
        ax2.set_ylim(y_min / PSI_TO_KPA, y_max / PSI_TO_KPA)
        ax2.set_ylabel("Overpressure [psi]", fontsize=10, alpha=0.7)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        if HAS_MPL:
            self.figure.clear()
            self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Threshold Table Tab
# ══════════════════════════════════════════════════════════════════════════════

class ThresholdTableTab(QWidget):
    """Distance-to-threshold table for each model."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        desc = QLabel(
            "Distances [m] at which specified overpressure thresholds are reached. "
            "Larger distances indicate greater hazard range."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(desc)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

    def display(self, results: Dict[str, Any]):
        """Populate threshold table from all results."""
        thresholds = [1.0, 3.0, 5.0, 8.0, 10.0]
        models = [k for k, v in results.items() if v is not None]

        if not models:
            self.clear()
            return

        # Setup columns: Threshold | Model1 | Model2 | Model3
        n_cols = 1 + len(models)
        self.table.setColumnCount(n_cols)
        headers = ["Threshold"] + [results[m].model_name.split(" (")[0] for m in models]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )

        self.table.setRowCount(len(thresholds))
        for i, psi in enumerate(thresholds):
            # Threshold label
            label = f"{psi} psi\n({psi * PSI_TO_KPA:.1f} kPa)"
            item = QTableWidgetItem(label)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.table.setItem(i, 0, item)

            # Distance per model
            for j, model_key in enumerate(models):
                result = results[model_key]
                if psi in result.distance_to_thresholds:
                    dist = result.distance_to_thresholds[psi]
                    item = QTableWidgetItem(f"{dist:.1f} m")
                else:
                    item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j + 1, item)

        self.table.resizeRowsToContents()

    def clear(self):
        self.table.setRowCount(0)
        self.table.setColumnCount(0)


# ══════════════════════════════════════════════════════════════════════════════
# Summary Tab
# ══════════════════════════════════════════════════════════════════════════════

class ExplosionSummaryTab(QWidget):
    """Summary of explosion analysis results."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(self.browser)

    def display(self, results: Dict[str, Any]):
        """Display HTML summary."""
        html_parts = [
            "<html><body style='font-family: sans-serif;'>",
            "<h2>💥 Explosion Analysis Summary</h2>",
        ]

        for model_key, result in results.items():
            if result is None:
                continue

            html_parts.append(f"<h3 style='color: {MODEL_COLORS.get(model_key.upper(), '#333')};'>")
            html_parts.append(f"{result.model_name}</h3>")

            html_parts.append("<table border='0' cellpadding='4' cellspacing='2'>")

            # Common metrics
            html_parts.append(
                f"<tr><td><b>Combustion Energy:</b></td>"
                f"<td>{result.energy/1e6:.1f} MJ "
                f"({result.energy/1e9:.2f} GJ)</td></tr>"
            )

            html_parts.append(
                f"<tr><td><b>TNT Equivalent:</b></td>"
                f"<td>{result.tnt_equivalent_mass:.1f} kg</td></tr>"
            )

            # Peak overpressure (at closest distance)
            min_dist = result.distances[0]
            max_op = result.overpressure[0]
            html_parts.append(
                f"<tr><td><b>Peak Overpressure at {min_dist:.0f} m:</b></td>"
                f"<td>{max_op:.1f} kPa ({max_op/PSI_TO_KPA:.1f} psi)</td></tr>"
            )

            # Model-specific params
            for key, value in result.model_params.items():
                label = key.replace("_", " ").title()
                if isinstance(value, float):
                    val_str = f"{value:.3f}"
                elif isinstance(value, int):
                    val_str = str(value)
                else:
                    val_str = str(value)
                html_parts.append(
                    f"<tr><td><b>{label}:</b></td><td>{val_str}</td></tr>"
                )

            html_parts.append("</table>")

            # Thresholds mini-table
            html_parts.append("<p><b>Damage Threshold Distances:</b></p>")
            html_parts.append(
                "<table border='1' cellpadding='3' cellspacing='0' "
                "style='border-collapse: collapse;'>"
            )
            html_parts.append(
                "<tr style='background-color: #E0E0E0;'>"
                "<th>Threshold</th><th>Distance</th>"
                "</tr>"
            )
            for psi in DAMAGE_THRESHOLDS_PSI:
                if psi in result.distance_to_thresholds:
                    d = result.distance_to_thresholds[psi]
                    html_parts.append(
                        f"<tr><td>{psi} psi ({psi*PSI_TO_KPA:.1f} kPa)</td>"
                        f"<td>{d:.1f} m</td></tr>"
                    )
            html_parts.append("</table>")
            html_parts.append("<br>")

        html_parts.append("</body></html>")
        self.browser.setHtml("".join(html_parts))

    def clear(self):
        self.browser.clear()
