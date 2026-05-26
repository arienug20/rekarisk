"""
Rekarisk UI — Fire Results Panel.

Displays fire calculation results using tables and matplotlib plots:
  - Thermal radiation vs distance plot
  - Distance to threshold table (4, 5, 12.5, 25, 37.5 kW/m²)
  - Flame geometry visualization (simple diagram)
  - Fireball visualization for BLEVE
  - Export CSV, export plot

Uses PyQt6 for the UI framework and matplotlib for plotting.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QSplitter, QScrollArea,
    QTextBrowser, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor

# matplotlib for plotting
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    import matplotlib
    matplotlib.use('Qt5Agg')


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

RADIATION_THRESHOLDS = [4.0, 5.0, 12.5, 25.0, 37.5]
THRESHOLD_LABELS = {
    4.0: "Solar radiation level",
    5.0: "Pain threshold (~20 s)",
    12.5: "1% lethality / significant injury",
    25.0: "Steel structure failure / wood ignition",
    37.5: "Process equipment damage / spontaneous ignition",
}
THRESHOLD_COLORS = {
    4.0: "#4CAF50",
    5.0: "#8BC34A",
    12.5: "#FFC107",
    25.0: "#FF9800",
    37.5: "#F44336",
}


# ══════════════════════════════════════════════════════════════════════════════
# Fire Results Panel
# ══════════════════════════════════════════════════════════════════════════════

class FireResultsPanel(QWidget):
    """Displays fire calculation results with plots and tables.

    Tabs:
      - Radiation: Thermal radiation vs distance plot
      - Thresholds: Distance to threshold table
      - Geometry: Flame geometry visualization
      - Summary: Text summary of results
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
        self.header_label = QLabel("Fire Results")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.header_label.setFont(font)
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        # Export buttons
        self.export_csv_btn = QPushButton("📥 Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_csv_btn.setEnabled(False)
        header_layout.addWidget(self.export_csv_btn)

        self.export_plot_btn = QPushButton("📊 Export Plot")
        self.export_plot_btn.clicked.connect(self._export_plot)
        self.export_plot_btn.setEnabled(False)
        header_layout.addWidget(self.export_plot_btn)

        layout.addLayout(header_layout)

        # Model info banner
        self.model_label = QLabel("")
        self.model_label.setStyleSheet(
            "background-color: #FFEBEE; padding: 6px; border-radius: 4px;"
        )
        self.model_label.setWordWrap(True)
        self.model_label.setVisible(False)
        layout.addWidget(self.model_label)

        # Tab widget
        self.tabs = QTabWidget()

        # Tab: Radiation Plot
        self._rad_tab = QWidget()
        rad_layout = QVBoxLayout(self._rad_tab)
        self.rad_figure = Figure(figsize=(8, 5), dpi=100)
        self.rad_canvas = FigureCanvas(self.rad_figure)
        rad_layout.addWidget(self.rad_canvas)
        self.tabs.addTab(self._rad_tab, "📈 Radiation")

        # Tab: Thresholds Table
        self._thresh_tab = QWidget()
        thresh_layout = QVBoxLayout(self._thresh_tab)
        self.thresh_table = QTableWidget()
        self.thresh_table.setColumnCount(4)
        self.thresh_table.setHorizontalHeaderLabels(
            ["Threshold [kW/m²]", "Distance [m]", "Description", "Likely Effect"]
        )
        self.thresh_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.thresh_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.thresh_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        thresh_layout.addWidget(self.thresh_table)
        self.tabs.addTab(self._thresh_tab, "📋 Thresholds")

        # Tab: Geometry Visualization
        self._geom_tab = QWidget()
        geom_layout = QVBoxLayout(self._geom_tab)
        self.geom_figure = Figure(figsize=(6, 6), dpi=100)
        self.geom_canvas = FigureCanvas(self.geom_figure)
        geom_layout.addWidget(self.geom_canvas)
        self.tabs.addTab(self._geom_tab, "🔷 Geometry")

        # Tab: Summary
        self._summary_tab = QWidget()
        summary_layout = QVBoxLayout(self._summary_tab)
        self.summary_browser = QTextBrowser()
        self.summary_browser.setOpenExternalLinks(True)
        summary_layout.addWidget(self.summary_browser)
        self.tabs.addTab(self._summary_tab, "📝 Summary")

        layout.addWidget(self.tabs)

        # Status messages
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; padding: 4px;")
        layout.addWidget(self.status_label)

    # ══════════════════════════════════════════════════════════════════════════
    # Display Methods
    # ══════════════════════════════════════════════════════════════════════════

    def display_pool_fire_result(self, result):
        """Display pool fire calculation results.

        Args:
            result: PoolFireResult from calculate_pool_fire().
        """
        self._model_type = "pool_fire"
        self._current_result = result
        self._update_display_common()
        self._plot_geometry_pool_fire(result)
        self._update_summary_pool_fire(result)

    def display_jet_fire_result(self, result):
        """Display jet fire calculation results.

        Args:
            result: JetFireResult from calculate_jet_fire().
        """
        self._model_type = "jet_fire"
        self._current_result = result
        self._update_display_common()
        self._plot_geometry_jet_fire(result)
        self._update_summary_jet_fire(result)

    def display_bleve_result(self, result):
        """Display BLEVE calculation results.

        Args:
            result: BLEVEResult from calculate_bleve().
        """
        self._model_type = "bleve"
        self._current_result = result
        self._update_display_common()
        self._plot_geometry_bleve(result)
        self._update_summary_bleve(result)

    def display_flash_fire_result(self, result):
        """Display flash fire calculation results.

        Args:
            result: FlashFireResult from calculate_flash_fire().
        """
        self._model_type = "flash_fire"
        self._current_result = result
        self._update_display_common()
        self._plot_geometry_flash_fire(result)
        self._update_summary_flash_fire(result)

    def _update_display_common(self):
        """Update plots and tables common to all fire types."""
        result = self._current_result
        if result is None:
            return

        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_plot_btn.setEnabled(True)

        # Update model banner
        model_names = {
            "pool_fire": "🔥 Pool Fire",
            "jet_fire": "💨 Jet Fire",
            "bleve": "💥 BLEVE / Fireball",
            "flash_fire": "🌫️ Flash Fire",
        }
        model_label = model_names.get(self._model_type, "Fire Model")
        self.model_label.setText(f"Model: {model_label}")
        self.model_label.setVisible(True)
        self.header_label.setText(f"{model_label} Results")

        # Plot radiation
        self._plot_radiation(result)

        # Update thresholds table
        self._update_thresholds_table(result)

        # Update status
        if hasattr(result, 'status_messages') and result.status_messages:
            self.status_label.setText(" | ".join(result.status_messages))

    def _plot_radiation(self, result):
        """Plot thermal radiation vs distance."""
        self.rad_figure.clear()
        ax = self.rad_figure.add_subplot(111)

        # Get radiation data
        rad_data = getattr(result, 'thermal_radiation_vs_distance', None)
        if rad_data is None or rad_data.shape[0] < 2:
            ax.text(0.5, 0.5, "No radiation data available",
                    ha='center', va='center', transform=ax.transAxes)
            self.rad_canvas.draw()
            return

        distances = rad_data[:, 0]
        fluxes = rad_data[:, 1]

        # Plot radiation curve
        ax.plot(distances, fluxes, 'r-', linewidth=2, label='Thermal Radiation')

        # Plot threshold lines
        thresholds = getattr(result, 'distance_to_thresholds', {})
        for thresh_val, thresh_dist in thresholds.items():
            if thresh_dist > 0 and thresh_dist < max(distances):
                color = THRESHOLD_COLORS.get(thresh_val, "#999")
                ax.axvline(x=thresh_dist, color=color, linestyle='--',
                          alpha=0.7, linewidth=1)
                ax.axhline(y=thresh_val, color=color, linestyle='--',
                          alpha=0.7, linewidth=1)
                ax.annotate(
                    f'{thresh_val} kW/m²\nat {thresh_dist} m',
                    xy=(thresh_dist, thresh_val),
                    xytext=(thresh_dist + 5, thresh_val + 2),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.7)
                )

        ax.set_xlabel('Distance from Fire [m]', fontsize=10)
        ax.set_ylabel('Thermal Radiation [kW/m²]', fontsize=10)
        ax.set_title(f'{self.header_label.text()} — Radiation vs Distance', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Use log scale for y if dynamic range is large
        max_flux = np.max(fluxes)
        if max_flux > 0 and fluxes[fluxes > 0].size > 0:
            min_flux = np.min(fluxes[fluxes > 0])
            if max_flux / max(min_flux, EPSILON := 0.001) > 100:
                ax.set_yscale('log')
                ax.set_ylabel('Thermal Radiation [kW/m²] (log)', fontsize=10)

        self.rad_figure.tight_layout()
        self.rad_canvas.draw()

    def _update_thresholds_table(self, result):
        """Update the thresholds table with calculation results."""
        thresholds = getattr(result, 'distance_to_thresholds', {})
        if not thresholds:
            self.thresh_table.setRowCount(0)
            return

        # Sort thresholds in descending order
        sorted_thresholds = sorted(thresholds.keys(), reverse=True)
        self.thresh_table.setRowCount(len(sorted_thresholds))

        effects = {
            37.5: "Spontaneous ignition of wood, process equipment damage",
            25.0: "Steel structure failure, piloted wood ignition (20 s)",
            12.5: "1% fatality (60 s), significant injury (30 s)",
            5.0: "Pain threshold (~20 s exposure)",
            4.0: "Safe for emergency personnel with protective clothing",
        }

        for i, thresh_val in enumerate(sorted_thresholds):
            dist = thresholds[thresh_val]
            label = THRESHOLD_LABELS.get(thresh_val, "")
            effect = effects.get(thresh_val, "")

            color = THRESHOLD_COLORS.get(thresh_val, "#EEE")

            # Threshold value
            item_val = QTableWidgetItem(f"{thresh_val:.1f}")
            item_val.setForeground(QColor(color))
            item_val.setFont(QFont("", weight=QFont.Weight.Bold))
            self.thresh_table.setItem(i, 0, item_val)

            # Distance
            item_dist = QTableWidgetItem(f"{dist:.1f}" if dist > 0 else "N/A")
            item_dist.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.thresh_table.setItem(i, 1, item_dist)

            # Description
            self.thresh_table.setItem(i, 2, QTableWidgetItem(label))

            # Effect
            self.thresh_table.setItem(i, 3, QTableWidgetItem(effect))

    # ══════════════════════════════════════════════════════════════════════════
    # Geometry Plots
    # ══════════════════════════════════════════════════════════════════════════

    def _plot_geometry_pool_fire(self, result):
        """Draw a simple diagram of pool fire geometry."""
        self.geom_figure.clear()
        ax = self.geom_figure.add_subplot(111)

        D = result.pool_diameter
        L = result.flame_length
        tilt = result.flame_tilt

        # Draw ground
        ax.axhline(y=0, color='#8B4513', linewidth=3, label='Ground')

        # Draw pool
        pool_rect = plt_Rectangle = None
        try:
            from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
        except ImportError:
            pass

        ax.add_patch(Rectangle((-D / 2, -0.1 * D), D, 0.05 * D,
                                facecolor='#654321', edgecolor='#3E2723', linewidth=1.5))
        ax.text(0, -0.3 * D, f'Pool D={D:.1f} m', ha='center', fontsize=9)

        # Draw flame as tilted rectangle/cylinder projection
        tilt_rad = np.radians(tilt)
        flame_x_end = L * np.sin(tilt_rad)
        flame_y_top = L * np.cos(tilt_rad)

        # Flame shape (tilted polygon)
        flame_x = [-D / 2 * np.cos(tilt_rad), D / 2 * np.cos(tilt_rad),
                   flame_x_end + D / 2, flame_x_end - D / 2]
        flame_y = [0, 0, flame_y_top, flame_y_top]

        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(
            list(zip(flame_x, flame_y)),
            facecolor='#FF5722', edgecolor='#BF360C',
            alpha=0.6, linewidth=2, label=f'Flame L={L:.1f} m'
        ))

        # Flame center annotation
        ax.annotate(f'L={L:.1f} m\ntilt={tilt:.0f}°',
                    xy=(flame_x_end / 2, flame_y_top / 2),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Horizontal [m]', fontsize=10)
        ax.set_ylabel('Height [m]', fontsize=10)
        ax.set_title('Pool Fire — Flame Geometry', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Set limits
        max_extent = max(D, L, flame_x_end) * 1.3
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-D * 0.4, flame_y_top * 1.2)

        self.geom_figure.tight_layout()
        self.geom_canvas.draw()

    def _plot_geometry_jet_fire(self, result):
        """Draw a simple diagram of jet fire geometry."""
        self.geom_figure.clear()
        ax = self.geom_figure.add_subplot(111)

        L = result.flame_length
        W = result.flame_width
        h_center = result.flame_center_height
        tilt = result.flame_tilt_deg

        # Ground
        ax.axhline(y=0, color='#8B4513', linewidth=3)

        # Release point
        ax.plot(0, 0, 'ko', markersize=8, label='Release Point')

        # Flame as cone/cylinder approximation
        tilt_rad = np.radians(tilt)

        # Draw flame shape
        from matplotlib.patches import Polygon, Circle

        # Base at release point
        # Flame body
        flame_x_start = 0
        flame_y_start = 0

        flame_x_end = L * np.sin(tilt_rad)
        flame_y_end = L * np.cos(tilt_rad)

        # Cone shape: narrow at base, wide at tip (or vice versa for momentum jet)
        n_points = 30
        t = np.linspace(0, 1, n_points)
        # Cone width increases linearly from 0.2W to W
        widths = 0.2 * W + 0.8 * W * t

        x_upper = t * flame_x_end + widths * np.cos(tilt_rad + np.pi / 2)
        y_upper = t * flame_y_end + widths * np.sin(tilt_rad + np.pi / 2)
        x_lower = t * flame_x_end - widths * np.cos(tilt_rad + np.pi / 2)
        y_lower = t * flame_y_end - widths * np.sin(tilt_rad + np.pi / 2)

        flame_x = np.concatenate([x_upper, x_lower[::-1]])
        flame_y = np.concatenate([y_upper, y_lower[::-1]])

        ax.add_patch(Polygon(
            list(zip(flame_x, flame_y)),
            facecolor='#FF5722', edgecolor='#BF360C',
            alpha=0.6, linewidth=2
        ))

        # Midpoint annotation
        ax.annotate(f'L={L:.1f} m\nW={W:.2f} m',
                    xy=(flame_x_end / 2, flame_y_end / 2 + W / 2),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Horizontal [m]', fontsize=10)
        ax.set_ylabel('Height [m]', fontsize=10)
        ax.set_title('Jet Fire — Flame Geometry', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        max_extent = max(L, W) * 1.3
        ax.set_xlim(-max_extent * 0.3, max_extent * 1.1)
        ax.set_ylim(-max_extent * 0.1, max_extent * 1.2)

        self.geom_figure.tight_layout()
        self.geom_canvas.draw()

    def _plot_geometry_bleve(self, result):
        """Draw a BLEVE fireball geometry diagram."""
        self.geom_figure.clear()
        ax = self.geom_figure.add_subplot(111)

        D = result.fireball_diameter
        R = D / 2.0
        h = result.center_height
        t_d = result.fireball_duration

        # Ground
        ax.axhline(y=0, color='#8B4513', linewidth=3, label='Ground')

        # Vessel marker at origin
        ax.plot(0, 0, 'ks', markersize=10, label='Vessel/Ground Zero')

        # Fireball sphere
        from matplotlib.patches import Circle
        fireball = Circle((0, h), R, facecolor='#FF9800', edgecolor='#E65100',
                         alpha=0.7, linewidth=3, label=f'Fireball D={D:.1f} m')
        ax.add_patch(fireball)

        # Radiative rays
        for angle in [0, 30, 60, 90, 120, 150]:
            rad = np.radians(angle)
            ray_start_x = (R + 2) * np.sin(rad)
            ray_start_y = h + (R + 2) * np.cos(rad)
            ray_end_x = (R + 10) * np.sin(rad)
            ray_end_y = h + (R + 10) * np.cos(rad)
            ax.arrow(ray_start_x, ray_start_y,
                    ray_end_x - ray_start_x, ray_end_y - ray_start_y,
                    head_width=1.5, head_length=2,
                    fc='red', ec='red', alpha=0.4, width=0.3)

        # Annotation
        ax.annotate(f'D={D:.1f} m\nduration={t_d:.1f} s\nSEP={result.sep:.0f} kW/m²',
                    xy=(0, h),
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # Fragment throw arrow (if available)
        if hasattr(result, 'fragment_max_distance') and result.fragment_max_distance > 0:
            frag_dist = result.fragment_max_distance
            ax.annotate('', xy=(frag_dist, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='red',
                                      linewidth=3, linestyle='--'))
            ax.text(frag_dist / 2, -2, f'Fragment throw:\n~{frag_dist:.0f} m',
                    ha='center', fontsize=8, color='red')

        ax.set_xlabel('Distance [m]', fontsize=10)
        ax.set_ylabel('Height [m]', fontsize=10)
        ax.set_title('BLEVE — Fireball Geometry', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        max_extent = max(D, h + R) * 1.3
        ax.set_xlim(-max_extent, max_extent * 1.2)
        ax.set_ylim(-2, h + R * 1.5)

        self.geom_figure.tight_layout()
        self.geom_canvas.draw()

    def _plot_geometry_flash_fire(self, result):
        """Draw flash fire LFL/UFL contour plot."""
        self.geom_figure.clear()
        ax = self.geom_figure.add_subplot(111)

        # Draw LFL contour
        if result.lfl_contour.shape[0] > 2:
            x = result.lfl_contour[:, 0]
            y = result.lfl_contour[:, 1]
            ax.fill(x, y, alpha=0.3, color='#FF9800', label=f'LFL (area={result.area_within_lfl:.0f} m²)')
            ax.plot(x, y, 'r-', linewidth=2, alpha=0.8)

        # Draw UFL contour
        if result.ufl_contour.shape[0] > 2:
            ux = result.ufl_contour[:, 0]
            uy = result.ufl_contour[:, 1]
            ax.fill(ux, uy, alpha=0.5, color='#F44336', label=f'UFL (area={result.area_within_ufl:.0f} m²)')
            ax.plot(ux, uy, 'r--', linewidth=1.5, alpha=0.6)

        # Source point
        ax.plot(0, 0, 'ko', markersize=8, label='Release Point')

        # Max distance annotation
        max_dist = result.max_distance_to_lfl
        if max_dist > 0:
            ax.annotate(f'Max LFL distance:\n{max_dist:.1f} m',
                        xy=(max_dist * 0.5, 0),
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Downwind Distance [m]', fontsize=10)
        ax.set_ylabel('Crosswind Distance [m]', fontsize=10)
        ax.set_title('Flash Fire — Flammable Cloud Envelope', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Set limits based on data
        if result.lfl_contour.shape[0] > 2:
            x_max = np.max(np.abs(result.lfl_contour[:, 0])) * 1.3
            y_max = np.max(np.abs(result.lfl_contour[:, 1])) * 1.3
            ax.set_xlim(-x_max * 0.2, x_max)
            ax.set_ylim(-y_max, y_max)
        else:
            ax.set_xlim(-5, 50)
            ax.set_ylim(-20, 20)

        self.geom_figure.tight_layout()
        self.geom_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Summary Updates
    # ══════════════════════════════════════════════════════════════════════════

    def _update_summary_pool_fire(self, result):
        """Generate text summary for pool fire results."""
        html = f"""
        <h2>🔥 Pool Fire Results</h2>
        <hr>
        <h3>Fire Characteristics</h3>
        <table border="0" cellpadding="4" cellspacing="2">
            <tr><td><b>Pool Diameter:</b></td><td>{result.pool_diameter:.2f} m</td></tr>
            <tr><td><b>Flame Length:</b></td><td>{result.flame_length:.2f} m</td></tr>
            <tr><td><b>Flame Tilt:</b></td><td>{result.flame_tilt:.1f}° from vertical</td></tr>
            <tr><td><b>Flame Drag:</b></td><td>{result.flame_drag:.2f} (downwind elongation)</td></tr>
            <tr><td><b>L/D Ratio:</b></td><td>{result.flame_length / max(result.pool_diameter, 0.001):.2f}</td></tr>
            <tr><td><b>Burning Rate:</b></td><td>{result.burning_rate * 1000:.2f} g/(m²·s)</td></tr>
            <tr><td><b>Total Burning Rate:</b></td><td>{result.total_burning_rate:.3f} kg/s</td></tr>
            <tr><td><b>Radiative Fraction:</b></td><td>{result.radiative_fraction:.3f}</td></tr>
            <tr><td><b>Surface Emissive Power:</b></td><td>{result.sep:.1f} kW/m²</td></tr>
            <tr><td><b>Model Used:</b></td><td>{result.model_used}</td></tr>
        </table>

        <h3>Threshold Distances</h3>
        <table border="0" cellpadding="4" cellspacing="2">
        """
        for thresh, dist in sorted(result.distance_to_thresholds.items(), reverse=True):
            label = THRESHOLD_LABELS.get(thresh, "")
            html += f"<tr><td><b>{thresh:.1f} kW/m²:</b></td><td>{dist:.1f} m</td><td><i>{label}</i></td></tr>"

        html += """
        </table>

        <h3>Interpretation</h3>
        <p>Pool fire thermal radiation depends on flame geometry (length, tilt),
        surface emissive power (SEP), atmospheric transmissivity, and the
        geometric view factor between the flame and target.</p>
        <p><b>Key distances to note:</b></p>
        <ul>
            <li>37.5 kW/m² — Spontaneous ignition of wood, process equipment damage</li>
            <li>12.5 kW/m² — 1% lethality for 60-second exposure</li>
            <li>4.0 kW/m² — Safe for emergency responders with protective clothing</li>
        </ul>
        """

        if result.status_messages:
            html += "<h3>Messages</h3><ul>"
            for msg in result.status_messages:
                html += f"<li>{msg}</li>"
            html += "</ul>"

        self.summary_browser.setHtml(html)

    def _update_summary_jet_fire(self, result):
        """Generate text summary for jet fire results."""
        html = f"""
        <h2>💨 Jet Fire Results</h2>
        <hr>
        <h3>Flame Characteristics</h3>
        <table border="0" cellpadding="4" cellspacing="2">
            <tr><td><b>Flame Length:</b></td><td>{result.flame_length:.2f} m</td></tr>
            <tr><td><b>Flame Width:</b></td><td>{result.flame_width:.3f} m</td></tr>
            <tr><td><b>Flame Center Height:</b></td><td>{result.flame_center_height:.2f} m</td></tr>
            <tr><td><b>Flame Tilt:</b></td><td>{result.flame_tilt_deg:.1f}°</td></tr>
            <tr><td><b>Total Heat Release:</b></td><td>{result.total_heat_release:.2f} MW</td></tr>
            <tr><td><b>Surface Emissive Power:</b></td><td>{result.sep:.1f} kW/m²</td></tr>
            <tr><td><b>Model Used:</b></td><td>{result.model_used}</td></tr>
        </table>

        <h3>Threshold Distances</h3>
        <table border="0" cellpadding="4" cellspacing="2">
        """
        for thresh, dist in sorted(result.distance_to_thresholds.items(), reverse=True):
            label = THRESHOLD_LABELS.get(thresh, "")
            html += f"<tr><td><b>{thresh:.1f} kW/m²:</b></td><td>{dist:.1f} m</td><td><i>{label}</i></td></tr>"

        html += """
        </table>

        <h3>Interpretation</h3>
        <p>Jet fire thermal radiation follows API RP 521 methodology.
        The flame is modeled as a cone-shaped turbulent diffusion flame,
        with the point source at 1/3 of the flame length from the release point.</p>
        <p>Jet fires can produce very high heat fluxes near the release point,
        but radiation drops rapidly with distance (1/R² dependence).</p>
        """

        if result.status_messages:
            html += "<h3>Messages</h3><ul>"
            for msg in result.status_messages:
                html += f"<li>{msg}</li>"
            html += "</ul>"

        self.summary_browser.setHtml(html)

    def _update_summary_bleve(self, result):
        """Generate text summary for BLEVE results."""
        html = f"""
        <h2>💥 BLEVE / Fireball Results</h2>
        <hr>
        <h3>Fireball Characteristics</h3>
        <table border="0" cellpadding="4" cellspacing="2">
            <tr><td><b>Fireball Diameter:</b></td><td>{result.fireball_diameter:.1f} m</td></tr>
            <tr><td><b>Fireball Duration:</b></td><td>{result.fireball_duration:.1f} s</td></tr>
            <tr><td><b>Center Height:</b></td><td>{result.center_height:.1f} m above ground</td></tr>
            <tr><td><b>Surface Emissive Power:</b></td><td>{result.sep:.0f} kW/m²</td></tr>
            <tr><td><b>Total Radiative Energy:</b></td><td>{result.total_radiative_energy:.2f} MJ</td></tr>
        """

        if result.fragment_max_distance > 0:
            html += f"""
            <tr><td><b>Fragment Max Distance:</b></td><td>≈ {result.fragment_max_distance:.0f} m (simplified)</td></tr>
            """

        html += """
        </table>

        <h3>Threshold Distances</h3>
        <table border="0" cellpadding="4" cellspacing="2">
        """
        for thresh, dist in sorted(result.distance_to_thresholds.items(), reverse=True):
            label = THRESHOLD_LABELS.get(thresh, "")
            html += f"<tr><td><b>{thresh:.1f} kW/m²:</b></td><td>{dist:.1f} m</td><td><i>{label}</i></td></tr>"

        html += """
        </table>

        <h3>Interpretation</h3>
        <p>BLEVE fireballs are short-duration, high-intensity thermal events.
        The Roberts correlation (D = 5.8 · M⁰·³³³) gives fireball dimensions,
        and the SEP typically ranges from 200-350 kW/m².</p>
        <p><b>Key characteristics:</b></p>
        <ul>
            <li>Short duration (seconds to tens of seconds)</li>
            <li>Very high thermal flux at close range</li>
            <li>Potential for fragment projectiles (missile effect)</li>
            <li>No overpressure effect from the fireball itself (separate from vessel rupture)</li>
        </ul>
        """

        if result.status_messages:
            html += "<h3>Messages</h3><ul>"
            for msg in result.status_messages:
                html += f"<li>{msg}</li>"
            html += "</ul>"

        self.summary_browser.setHtml(html)

    def _update_summary_flash_fire(self, result):
        """Generate text summary for flash fire results."""
        has_lfl = result.lfl_contour.shape[0] > 2

        html = f"""
        <h2>🌫️ Flash Fire Results</h2>
        <hr>
        <h3>Flammable Cloud Characteristics</h3>
        <table border="0" cellpadding="4" cellspacing="2">
            <tr><td><b>Area within LFL:</b></td><td>{result.area_within_lfl:.0f} m²</td></tr>
            <tr><td><b>Area within UFL:</b></td><td>{result.area_within_ufl:.0f} m²</td></tr>
            <tr><td><b>Max Distance to LFL:</b></td><td>{result.max_distance_to_lfl:.1f} m</td></tr>
            <tr><td><b>LFL Contour Points:</b></td><td>{result.lfl_contour.shape[0]}</td></tr>
            <tr><td><b>UFL Contour Points:</b></td><td>{result.ufl_contour.shape[0]}</td></tr>
        </table>

        <h3>Interpretation</h3>
        """

        if has_lfl:
            html += f"""
        <p>A flammable cloud extending approximately {result.max_distance_to_lfl:.1f} m
        from the release point has been identified. The area within the LFL is
        {result.area_within_lfl:.0f} m², representing the zone where a flash fire
        could occur if ignited.</p>
        <p><b>Inside the LFL contour:</b> Potential for flash fire — personnel
        should not be present without appropriate PPE.</p>
        <p><b>Outside the LFL contour:</b> Concentration is below the lower
        flammable limit, so flash fire is not possible.</p>
            """
        else:
            html += """
        <p><b>No LFL contour was found</b> — the concentration field may not
        have reached the lower flammable limit at any grid point. This could
        mean the release rate is too low, or the grid does not extend far
        enough toward the source.</p>
        <p>Consider:
        <ul>
            <li>Increasing the release rate</li>
            <li>Using a finer grid near the source</li>
            <li>Checking that the substance's LFL is correct</li>
        </ul>
        </p>
            """

        html += """
        <h3>Flash Fire Effects</h3>
        <p>Flash fires are short-duration combustion of a flammable vapor cloud.
        The hazard zone is the LFL isopleth — ignition inside this zone can cause
        a fire that propagates through the cloud.</p>
        <ul>
            <li>Personnel within the LFL zone → potentially fatal</li>
            <li>Flash fire SEP typically 150-200 kW/m² (short duration)</li>
            <li>Unlike pool/jet fires, flash fires have no defined flame geometry</li>
        </ul>
        """

        if result.status_messages:
            html += "<h3>Messages</h3><ul>"
            for msg in result.status_messages:
                html += f"<li>{msg}</li>"
            html += "</ul>"

        self.summary_browser.setHtml(html)

    # ══════════════════════════════════════════════════════════════════════════
    # Export Functions
    # ══════════════════════════════════════════════════════════════════════════

    def _export_csv(self):
        """Export thermal radiation data to CSV file."""
        result = self._current_result
        if result is None:
            return

        rad_data = getattr(result, 'thermal_radiation_vs_distance', None)
        if rad_data is None or rad_data.shape[0] < 2:
            QMessageBox.warning(self, "Export Error",
                               "No radiation data available to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "fire_radiation.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Distance [m]", "Thermal Radiation [kW/m2]"])
                for row in rad_data:
                    writer.writerow([f"{row[0]:.3f}", f"{row[1]:.6f}"])

            QMessageBox.information(self, "Export Complete",
                                   f"Data exported to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_plot(self):
        """Export the current radiation plot to an image file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "fire_radiation_plot.png",
            "PNG Images (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if not filepath:
            return

        try:
            self.rad_figure.savefig(filepath, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, "Export Complete",
                                   f"Plot saved to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def clear(self):
        """Clear all displayed results."""
        self._current_result = None
        self._model_type = None

        self.rad_figure.clear()
        self.rad_canvas.draw()

        self.geom_figure.clear()
        self.geom_canvas.draw()

        self.thresh_table.setRowCount(0)
        self.summary_browser.clear()
        self.model_label.setVisible(False)
        self.header_label.setText("Fire Results")
        self.status_label.setText("")

        self.export_csv_btn.setEnabled(False)
        self.export_plot_btn.setEnabled(False)
