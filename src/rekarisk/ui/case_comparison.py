"""
Rekarisk UI — Case Comparison Panel.

PyQt6 panel for multi-case overlay and comparison:
    - Scenario list with checkboxes (select cases to compare)
    - Overlay plot: multiple contours on same canvas
    - Comparison table: key metrics side by side
    - Difference map: scenario A minus scenario B
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QGroupBox,
    QScrollArea, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import cm
    import matplotlib
    matplotlib.use("Qt5Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Case Comparison Panel
# ══════════════════════════════════════════════════════════════════════════════

class CaseComparisonPanel(QWidget):
    """Multi-case overlay and comparison panel.

    Layout:
      ┌──────────────────────────────────────────────────┐
      │  Scenario Selection (checkboxes)                 │
      ├─────────────────────┬────────────────────────────┤
      │                     │                            │
      │   Overlay Plot      │   Comparison Table         │
      │   (Matplotlib)      │   (QTableWidget)           │
      │                     │                            │
      ├─────────────────────┴────────────────────────────┤
      │  Difference Map Controls (scenario A - B)        │
      └──────────────────────────────────────────────────┘

    Signals:
        comparison_updated: Emitted when comparison data changes.
    """

    comparison_updated = pyqtSignal(list)  # list of selected result names

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._all_results: List[Dict[str, Any]] = []
        self._checkboxes: List[QCheckBox] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Scenario Selection ──
        sel_group = QGroupBox("Select Scenarios to Compare")
        sel_layout = QHBoxLayout(sel_group)
        self._check_container = QWidget()
        self._check_layout = QVBoxLayout(self._check_container)
        self._check_layout.setSpacing(2)
        self._check_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidget(self._check_container)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(120)
        sel_layout.addWidget(scroll)

        # Select all / none
        btn_layout = QVBoxLayout()
        select_all_btn = QPushButton("All")
        select_all_btn.setMaximumWidth(50)
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton("None")
        select_none_btn.setMaximumWidth(50)
        select_none_btn.clicked.connect(self._select_none)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addStretch()
        sel_layout.addLayout(btn_layout)

        layout.addWidget(sel_group)

        # ── Main splitter: plot | table ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Overlay Plot
        if HAS_MPL:
            self._figure = Figure(figsize=(6, 5), facecolor="white")
            self._canvas = FigureCanvas(self._figure)
            self._ax = self._figure.add_subplot(111)
            self._ax.set_xlabel("X (m)")
            self._ax.set_ylabel("Y (m)")
            self._ax.set_title("Multi-Case Overlay", fontsize=12, fontweight="bold")
            self._ax.grid(True, alpha=0.3, linestyle="--")
            self._ax.set_aspect("equal")
            self._figure.tight_layout()
            plot_widget = self._canvas
        else:
            plot_widget = QLabel("Matplotlib not available for overlay plot.")
            plot_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        splitter.addWidget(plot_widget)

        # Right: Comparison Table
        table_group = QGroupBox("Comparison Table")
        table_layout = QVBoxLayout(table_group)

        self._comp_table = QTableWidget()
        self._comp_table.setAlternatingRowColors(True)
        self._comp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._comp_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self._comp_table)

        splitter.addWidget(table_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

        # ── Difference Map ──
        diff_group = QGroupBox("Difference Map (Scenario A − Scenario B)")
        diff_layout = QHBoxLayout(diff_group)

        diff_layout.addWidget(QLabel("Scenario A:"))
        self._diff_a_combo = QComboBox()
        self._diff_a_combo.setMinimumWidth(200)
        diff_layout.addWidget(self._diff_a_combo)

        diff_layout.addWidget(QLabel("Scenario B:"))
        self._diff_b_combo = QComboBox()
        self._diff_b_combo.setMinimumWidth(200)
        diff_layout.addWidget(self._diff_b_combo)

        compute_diff_btn = QPushButton("Compute Difference")
        compute_diff_btn.clicked.connect(self._compute_difference)
        diff_layout.addWidget(compute_diff_btn)

        clear_diff_btn = QPushButton("Clear")
        clear_diff_btn.clicked.connect(self._clear_difference)
        diff_layout.addWidget(clear_diff_btn)

        layout.addWidget(diff_group)

        # ── Difference Plot (bottom) ──
        if HAS_MPL:
            self._diff_figure = Figure(figsize=(8, 3), facecolor="white")
            self._diff_canvas = FigureCanvas(self._diff_figure)
            self._diff_ax = self._diff_figure.add_subplot(111)
            self._diff_ax.set_xlabel("X (m)")
            self._diff_ax.set_ylabel("Y (m)")
            self._diff_ax.set_title("Difference Map", fontsize=11, fontweight="bold")
            self._diff_ax.grid(True, alpha=0.3, linestyle="--")
            self._diff_figure.tight_layout()
            diff_plot = self._diff_canvas
        else:
            diff_plot = QLabel("Matplotlib not available.")
            diff_plot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(diff_plot)

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def set_results(self, results: List[Dict[str, Any]]):
        """Load results for comparison.

        Parameters
        ----------
        results : list[dict]
            List of result dicts, each with at least "name", "type", "summary".
        """
        self._all_results = results
        self._build_checkboxes()
        self._update_combos()
        self._update_plot_and_table()

    def add_result(self, result: Dict[str, Any]):
        """Add a single result to the comparison."""
        self._all_results.append(result)
        self._build_checkboxes()
        self._update_combos()

    def clear(self):
        """Clear all data."""
        self._all_results = []
        self._build_checkboxes()
        self._update_combos()
        self._clear_plot()
        self._comp_table.clear()
        self._comp_table.setRowCount(0)
        self._comp_table.setColumnCount(0)

    # ══════════════════════════════════════════════════════════════════════════
    # Internal — Checkboxes
    # ══════════════════════════════════════════════════════════════════════════

    def _build_checkboxes(self):
        """Rebuild checkbox list from results."""
        # Clear old
        for cb in self._checkboxes:
            self._check_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        for res in self._all_results:
            name = res.get("name", "Unnamed")
            cb = QCheckBox(f"{name}  [{res.get('type', '?')}]")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda s, r=res: self._update_plot_and_table())
            self._check_layout.addWidget(cb)
            self._checkboxes.append(cb)

    def _get_selected_results(self) -> List[Dict[str, Any]]:
        """Return list of currently checked results."""
        selected = []
        for cb, res in zip(self._checkboxes, self._all_results):
            if cb.isChecked():
                selected.append(res)
        return selected

    def _select_all(self):
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _select_none(self):
        for cb in self._checkboxes:
            cb.setChecked(False)

    # ══════════════════════════════════════════════════════════════════════════
    # Internal — Combos for difference map
    # ══════════════════════════════════════════════════════════════════════════

    def _update_combos(self):
        """Update combo boxes with result names."""
        self._diff_a_combo.clear()
        self._diff_b_combo.clear()
        for res in self._all_results:
            name = res.get("name", "Unnamed")
            self._diff_a_combo.addItem(name)
            self._diff_b_combo.addItem(name)

    # ══════════════════════════════════════════════════════════════════════════
    # Internal — Plot & Table
    # ══════════════════════════════════════════════════════════════════════════

    def _update_plot_and_table(self):
        """Update overlay plot and comparison table from selected results."""
        selected = self._get_selected_results()
        if not selected:
            self._clear_plot()
            self._comp_table.clear()
            self._comp_table.setRowCount(0)
            return

        self._update_overlay_plot(selected)
        self._update_comparison_table(selected)
        self.comparison_updated.emit([r.get("name", "") for r in selected])

    def _update_overlay_plot(self, selected: List[Dict[str, Any]]):
        """Draw multiple contours on the overlay axes."""
        if not HAS_MPL:
            return

        self._ax.clear()
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_title("Multi-Case Overlay", fontsize=12, fontweight="bold")
        self._ax.grid(True, alpha=0.3, linestyle="--")

        import matplotlib
        colors = matplotlib.cm.tab10.colors if hasattr(matplotlib.cm, 'tab10') else [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ]

        for i, res in enumerate(selected):
            color = colors[i % len(colors)]
            name = res.get("name", f"Case {i+1}")
            grid = res.get("grid_data", res)

            if "x" in grid and "y" in grid and "Z" in grid:
                x = np.asarray(grid["x"], dtype=float)
                y = np.asarray(grid["y"], dtype=float)
                Z = np.asarray(grid["Z"], dtype=float)
                levels = grid.get("levels", [])

                if Z.ndim == 2 and len(levels) > 0:
                    # Draw outermost contour
                    try:
                        self._ax.contour(x, y, Z, levels=[levels[0]],
                                         colors=[color], linewidths=2,
                                         label=name)
                    except Exception:
                        pass

                    # Try filled low-alpha
                    try:
                        self._ax.contourf(x, y, Z, levels=[levels[0], levels[-1]],
                                          colors=[color], alpha=0.08)
                    except Exception:
                        pass

            # Check for pre-computed contour segments
            contours = res.get("contours", res.get("contour", []))
            if isinstance(contours, list) and contours:
                for seg in contours:
                    if isinstance(seg, (list, np.ndarray)) and len(seg) >= 2:
                        pts = np.asarray(seg)
                        if pts.ndim == 2 and pts.shape[1] >= 2:
                            self._ax.plot(pts[:, 0], pts[:, 1], color=color,
                                          linewidth=1.5, alpha=0.8)

        # Source marker
        self._ax.plot(0, 0, "k*", markersize=10, label="Source", zorder=10)
        self._ax.legend(loc="upper right", fontsize=8, ncol=1)
        self._ax.set_aspect("equal")
        self._figure.tight_layout()
        self._canvas.draw()

    def _update_comparison_table(self, selected: List[Dict[str, Any]]):
        """Populate the comparison table with key metrics."""

        # Gather all unique metric names
        all_keys: List[str] = []
        for res in selected:
            summary = res.get("summary", {})
            for k in summary:
                if k not in all_keys:
                    all_keys.append(k)

        if not all_keys:
            all_keys = ["No metrics available"]

        # Build table
        self._comp_table.clear()
        self._comp_table.setRowCount(len(all_keys))
        self._comp_table.setColumnCount(1 + len(selected))

        # Headers
        self._comp_table.setHorizontalHeaderLabels(
            ["Parameter"] + [res.get("name", f"Case {i+1}")[:25]
                             for i, res in enumerate(selected)]
        )

        # Fill data
        for row, key in enumerate(all_keys):
            item = QTableWidgetItem(str(key))
            item.setFont(QFont("", -1, QFont.Weight.Bold))
            self._comp_table.setItem(row, 0, item)

            for col, res in enumerate(selected):
                summary = res.get("summary", {})
                value = summary.get(key, "—")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._comp_table.setItem(row, col + 1, item)

        # Resize
        self._comp_table.resizeColumnsToContents()
        self._comp_table.horizontalHeader().setStretchLastSection(True)
        self._comp_table.verticalHeader().setVisible(False)

    def _clear_plot(self):
        """Clear the overlay plot."""
        if HAS_MPL:
            self._ax.clear()
            self._ax.set_xlabel("X (m)")
            self._ax.set_ylabel("Y (m)")
            self._ax.set_title("Multi-Case Overlay", fontsize=12, fontweight="bold")
            self._ax.grid(True, alpha=0.3, linestyle="--")
            self._ax.set_aspect("equal")
            self._figure.tight_layout()
            self._canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Internal — Difference Map
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_difference(self):
        """Compute and plot the difference between two scenarios."""
        if not HAS_MPL:
            return

        idx_a = self._diff_a_combo.currentIndex()
        idx_b = self._diff_b_combo.currentIndex()

        if idx_a < 0 or idx_b < 0 or idx_a >= len(self._all_results) or idx_b >= len(self._all_results):
            return

        res_a = self._all_results[idx_a]
        res_b = self._all_results[idx_b]

        grid_a = res_a.get("grid_data", res_a)
        grid_b = res_b.get("grid_data", res_b)

        if "Z" not in grid_a or "Z" not in grid_b:
            QMessageBox.warning(self, "No Data",
                                "One or both scenarios do not have gridded data for difference computation.")
            return

        Za = np.asarray(grid_a["Z"], dtype=float)
        Zb = np.asarray(grid_b["Z"], dtype=float)

        if Za.shape != Zb.shape:
            # Try to interpolate to common grid
            try:
                from scipy.interpolate import griddata
                x = np.asarray(grid_a.get("x", []), dtype=float)
                y = np.asarray(grid_a.get("y", []), dtype=float)
                xb = np.asarray(grid_b.get("x", []), dtype=float)
                yb = np.asarray(grid_b.get("y", []), dtype=float)
                Xb, Yb = np.meshgrid(xb, yb)
                points = np.column_stack((Xb.ravel(), Yb.ravel()))
                values = Zb.ravel()
                Xa, Ya = np.meshgrid(x, y)
                Zb_interp = griddata(points, values, (Xa, Ya), method="linear", fill_value=0)
                Z_diff = Za - Zb_interp
                x_plot = x
                y_plot = y
            except Exception:
                QMessageBox.warning(self, "Grid Mismatch",
                                    "Cannot compute difference: grids have different shapes.")
                return
        else:
            Z_diff = Za - Zb
            x_plot = np.asarray(grid_a.get("x", []), dtype=float)
            y_plot = np.asarray(grid_a.get("y", []), dtype=float)

        name_a = res_a.get("name", "A")
        name_b = res_b.get("name", "B")

        # Plot
        self._diff_ax.clear()
        self._diff_ax.set_xlabel("X (m)")
        self._diff_ax.set_ylabel("Y (m)")
        self._diff_ax.set_title(f"Difference: {name_a} − {name_b}", fontsize=11, fontweight="bold")
        self._diff_ax.grid(True, alpha=0.3, linestyle="--")

        # Use RdBu-like colormap centered at 0
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        vmax = max(abs(Z_diff.min()), abs(Z_diff.max()), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        try:
            cf = self._diff_ax.pcolormesh(x_plot, y_plot, Z_diff,
                                          cmap="RdBu_r", norm=norm, shading="auto")
            cbar = self._diff_figure.colorbar(cf, ax=self._diff_ax,
                                              label="Difference", shrink=0.85)
            self._diff_ax.plot(0, 0, "k*", markersize=10, zorder=10)
            self._diff_ax.set_aspect("equal")
            self._diff_figure.tight_layout()
            self._diff_canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))

    def _clear_difference(self):
        """Clear the difference plot."""
        if HAS_MPL:
            self._diff_ax.clear()
            self._diff_ax.set_xlabel("X (m)")
            self._diff_ax.set_ylabel("Y (m)")
            self._diff_ax.set_title("Difference Map", fontsize=11, fontweight="bold")
            self._diff_ax.grid(True, alpha=0.3, linestyle="--")
            self._diff_figure.tight_layout()
            self._diff_canvas.draw()
