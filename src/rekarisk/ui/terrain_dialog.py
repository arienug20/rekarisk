"""
Rekarisk UI — Terrain & Obstacle Configuration Dialog.

Provides a three-tab dialog for:
    1. Obstacles  — Table-based obstacle editor (add, remove, edit properties)
    2. DEM        — Digital Elevation Model CSV import and preview
    3. LOS Preview — Line-of-sight diagram with source, obstacles, receptor

Uses PyQt6. QAction is in PyQt6.QtGui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import (
    QAction, QColor, QDoubleValidator, QFont, QPainter, QPen, QBrush,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..terrain.obstacle import (
    Obstacle, ObstacleCollection, OBSTACLE_TYPES, OBSTACLE_TYPE_LABELS,
)
from ..terrain.dem_loader import DEMData
from ..terrain.los_engine import LOSEngine, LOSResult, LOSStatus, SourceGeometry


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle Tab
# ══════════════════════════════════════════════════════════════════════════════

class ObstacleTab(QWidget):
    """Tab for managing terrain obstacles in a table view."""

    obstacles_changed = pyqtSignal()

    _COLUMNS = [
        "Name", "Type", "X [m]", "Y [m]",
        "Length [m]", "Width [m]", "Height [m]",
        "Orient [°]", "Porosity",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collection = ObstacleCollection()
        self._setup_ui()

    # -- Collection access ----------------------------------------------------

    @property
    def collection(self) -> ObstacleCollection:
        return self._collection

    @collection.setter
    def collection(self, col: ObstacleCollection) -> None:
        self._collection = col
        self._populate_table()

    def set_collection(self, col: ObstacleCollection) -> None:
        self._collection = col
        self._populate_table()

    # -- UI setup -------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        self._btn_add = QPushButton("➕ Add Obstacle")
        self._btn_add.clicked.connect(self._add_obstacle)
        toolbar.addWidget(self._btn_add)

        self._btn_remove = QPushButton("➖ Remove Selected")
        self._btn_remove.clicked.connect(self._remove_selected)
        toolbar.addWidget(self._btn_remove)

        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self._clear_all)
        toolbar.addWidget(self._btn_clear)

        toolbar.addStretch()

        self._btn_import = QPushButton("Import JSON...")
        self._btn_import.clicked.connect(self._import_json)
        toolbar.addWidget(self._btn_import)

        self._btn_export = QPushButton("Export JSON...")
        self._btn_export.clicked.connect(self._export_json)
        toolbar.addWidget(self._btn_export)

        layout.addLayout(toolbar)

        # ── Table ──
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self._table.setColumnWidth(0, 120)
        self._table.setAlternatingRowColors(True)

        layout.addWidget(self._table)

        # ── Quick-add form ──
        group = QGroupBox("Quick Add")
        form = QFormLayout()

        self._quick_name = QLineEdit()
        self._quick_name.setPlaceholderText("e.g., Control Room")
        form.addRow("Name:", self._quick_name)

        self._quick_type = QComboBox()
        for t in OBSTACLE_TYPES:
            self._quick_type.addItem(OBSTACLE_TYPE_LABELS.get(t, t), t)
        form.addRow("Type:", self._quick_type)

        pos_layout = QHBoxLayout()
        self._quick_x = QLineEdit("0.0")
        self._quick_x.setValidator(QDoubleValidator(-99999, 99999, 2))
        self._quick_x.setMaximumWidth(80)
        pos_layout.addWidget(QLabel("X [m]:"))
        pos_layout.addWidget(self._quick_x)
        self._quick_y = QLineEdit("0.0")
        self._quick_y.setValidator(QDoubleValidator(-99999, 99999, 2))
        self._quick_y.setMaximumWidth(80)
        pos_layout.addWidget(QLabel("Y [m]:"))
        pos_layout.addWidget(self._quick_y)
        pos_layout.addStretch()
        form.addRow("Position:", pos_layout)

        dim_layout = QHBoxLayout()
        self._quick_l = QLineEdit("10.0")
        self._quick_l.setValidator(QDoubleValidator(0.01, 9999, 2))
        self._quick_l.setMaximumWidth(70)
        dim_layout.addWidget(QLabel("L [m]:"))
        dim_layout.addWidget(self._quick_l)
        self._quick_w = QLineEdit("10.0")
        self._quick_w.setValidator(QDoubleValidator(0.01, 9999, 2))
        self._quick_w.setMaximumWidth(70)
        dim_layout.addWidget(QLabel("W [m]:"))
        dim_layout.addWidget(self._quick_w)
        self._quick_h = QLineEdit("5.0")
        self._quick_h.setValidator(QDoubleValidator(0.01, 9999, 2))
        self._quick_h.setMaximumWidth(70)
        dim_layout.addWidget(QLabel("H [m]:"))
        dim_layout.addWidget(self._quick_h)
        dim_layout.addStretch()
        form.addRow("Dimensions:", dim_layout)

        self._quick_orient = QLineEdit("0.0")
        self._quick_orient.setValidator(QDoubleValidator(0.0, 360.0, 2))
        self._quick_orient.setMaximumWidth(80)
        form.addRow("Orientation [°]:", self._quick_orient)

        self._quick_porosity = QLineEdit("0.0")
        self._quick_porosity.setValidator(QDoubleValidator(0.0, 1.0, 3))
        self._quick_porosity.setMaximumWidth(80)
        form.addRow("Porosity:", self._quick_porosity)

        btn_row = QHBoxLayout()
        self._btn_quick_add = QPushButton("Add to Table")
        self._btn_quick_add.clicked.connect(self._quick_add)
        btn_row.addWidget(self._btn_quick_add)
        btn_row.addStretch()
        form.addRow(btn_row)

        group.setLayout(form)
        layout.addWidget(group)

        # ── Summary label ──
        self._summary_label = QLabel("No obstacles defined.")
        self._summary_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._summary_label)

    # -- Slot: add obstacle ---------------------------------------------------

    def _add_obstacle(self) -> None:
        """Add a default obstacle and scroll to it."""
        obs = Obstacle(
            name=f"Obstacle {len(self._collection) + 1}",
            type="building",
            position=(0.0, 0.0),
            dimensions=(10.0, 10.0, 5.0),
        )
        self._collection.add(obs)
        self._append_table_row(obs)
        self._update_summary()
        self.obstacles_changed.emit()

    def _quick_add(self) -> None:
        """Add obstacle from the quick-add form."""
        name = self._quick_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a name.")
            return

        try:
            x = float(self._quick_x.text())
            y = float(self._quick_y.text())
            l = float(self._quick_l.text())
            w = float(self._quick_w.text())
            h = float(self._quick_h.text())
            orient = float(self._quick_orient.text())
            poro = float(self._quick_porosity.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers.")
            return

        obs_type = self._quick_type.currentData()
        obs = Obstacle(
            name=name,
            type=obs_type,
            position=(x, y),
            dimensions=(l, w, h),
            orientation=orient,
            porosity=poro,
        )
        self._collection.add(obs)
        self._append_table_row(obs)
        self._update_summary()
        self.obstacles_changed.emit()

    def _remove_selected(self) -> None:
        """Remove the currently selected obstacle."""
        current = self._table.currentRow()
        if current < 0 or current >= len(self._collection):
            return
        obs = self._collection[current]
        self._collection.remove(obs.id)
        self._table.removeRow(current)
        self._update_summary()
        self.obstacles_changed.emit()

    def _clear_all(self) -> None:
        """Remove all obstacles."""
        if not self._collection.obstacles:
            return
        reply = QMessageBox.question(
            self, "Clear All Obstacles",
            "Remove all obstacles? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._collection.clear()
            self._table.setRowCount(0)
            self._update_summary()
            self.obstacles_changed.emit()

    # -- Import/Export --------------------------------------------------------

    def _import_json(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Obstacles from JSON", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return
        try:
            col = ObstacleCollection.from_json(filepath)
            self._collection = col
            self._populate_table()
            self._update_summary()
            self.obstacles_changed.emit()
        except Exception as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Failed to import obstacles:\n{e}",
            )

    def _export_json(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Obstacles to JSON", "obstacles.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return
        try:
            self._collection.to_json(filepath)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(self._collection)} obstacles to:\n{filepath}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export:\n{e}",
            )

    # -- Table population -----------------------------------------------------

    def _populate_table(self) -> None:
        """Fill the table from the current collection."""
        self._table.setRowCount(0)
        for obs in self._collection:
            self._append_table_row(obs)
        self._update_summary()

    def _append_table_row(self, obs: Obstacle) -> None:
        """Add a single row for an obstacle."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        cells = [
            obs.name,
            OBSTACLE_TYPE_LABELS.get(obs.type, obs.type),
            f"{obs.x:.1f}",
            f"{obs.y:.1f}",
            f"{obs.length:.1f}",
            f"{obs.width:.1f}",
            f"{obs.height:.1f}",
            f"{obs.orientation:.1f}",
            f"{obs.porosity:.2f}",
        ]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, obs.id)
            if col >= 2:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            self._table.setItem(row, col, item)

    def _update_summary(self) -> None:
        n = len(self._collection)
        if n == 0:
            self._summary_label.setText("No obstacles defined.")
        else:
            summary = self._collection.type_summary()
            types_str = ", ".join(
                f"{OBSTACLE_TYPE_LABELS.get(k,k)}: {v}" for k, v in summary.items()
            )
            area = self._collection.total_footprint_area()
            max_h = self._collection.max_height()
            self._summary_label.setText(
                f"{n} obstacle(s) | Types: {types_str} | "
                f"Total footprint: {area:.0f} m² | Max height: {max_h:.1f} m"
            )


# ══════════════════════════════════════════════════════════════════════════════
# DEM Tab
# ══════════════════════════════════════════════════════════════════════════════

class DEMTab(QWidget):
    """Tab for loading and inspecting Digital Elevation Models."""

    dem_loaded = pyqtSignal(object)  # emits DEMData

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dem: DEMData | None = None
        self._setup_ui()

    @property
    def dem(self) -> DEMData | None:
        return self._dem

    # -- UI setup -------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Load controls ──
        load_group = QGroupBox("Load DEM")
        load_layout = QFormLayout()

        file_layout = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("Select CSV file (x, y, z columns)...")
        self._file_path.setReadOnly(True)
        file_layout.addWidget(self._file_path)
        self._btn_browse = QPushButton("Browse...")
        self._btn_browse.clicked.connect(self._browse_csv)
        file_layout.addWidget(self._btn_browse)
        load_layout.addRow("CSV File:", file_layout)

        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("X column:"))
        self._x_col = QSpinBox()
        self._x_col.setRange(0, 99)
        self._x_col.setValue(0)
        col_layout.addWidget(self._x_col)
        col_layout.addWidget(QLabel("Y column:"))
        self._y_col = QSpinBox()
        self._y_col.setRange(0, 99)
        self._y_col.setValue(1)
        col_layout.addWidget(self._y_col)
        col_layout.addWidget(QLabel("Z column:"))
        self._z_col = QSpinBox()
        self._z_col.setRange(0, 99)
        self._z_col.setValue(2)
        col_layout.addWidget(self._z_col)
        col_layout.addStretch()
        load_layout.addRow("Columns:", col_layout)

        delim_layout = QHBoxLayout()
        self._delim_comma = QCheckBox("Comma (,)")
        self._delim_comma.setChecked(True)
        delim_layout.addWidget(self._delim_comma)
        self._delim_other = QCheckBox("Other:")
        delim_layout.addWidget(self._delim_other)
        self._delim_text = QLineEdit(";")
        self._delim_text.setMaximumWidth(40)
        delim_layout.addWidget(self._delim_text)
        delim_layout.addStretch()
        load_layout.addRow("Delimiter:", delim_layout)

        self._has_header = QCheckBox("First row is header")
        load_layout.addRow("", self._has_header)

        self._btn_load = QPushButton("Load DEM")
        self._btn_load.clicked.connect(self._load_csv)
        load_layout.addRow(self._btn_load)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # ── Stats / Info ──
        self._info_group = QGroupBox("DEM Information")
        self._info_layout = QFormLayout()
        self._info_labels: Dict[str, QLabel] = {}

        for key in ["Name", "Grid Shape", "Cell Size", "X Range", "Y Range",
                     "Z Min", "Z Max", "Z Range"]:
            label = QLabel("—")
            label.setStyleSheet("color: gray;")
            self._info_labels[key] = label
            self._info_layout.addRow(f"{key}:", label)

        self._info_group.setLayout(self._info_layout)
        layout.addWidget(self._info_group)

        # ── Elevation Query ──
        query_group = QGroupBox("Elevation Query")
        query_layout = QFormLayout()

        q_coord = QHBoxLayout()
        q_coord.addWidget(QLabel("X [m]:"))
        self._query_x = QLineEdit("0.0")
        self._query_x.setValidator(QDoubleValidator(-99999, 99999, 2))
        self._query_x.setMaximumWidth(100)
        q_coord.addWidget(self._query_x)
        q_coord.addWidget(QLabel("Y [m]:"))
        self._query_y = QLineEdit("0.0")
        self._query_y.setValidator(QDoubleValidator(-99999, 99999, 2))
        self._query_y.setMaximumWidth(100)
        q_coord.addWidget(self._query_y)
        q_coord.addStretch()
        query_layout.addRow("Point:", q_coord)

        self._btn_query = QPushButton("Query Elevation")
        self._btn_query.clicked.connect(self._query_elevation)
        query_layout.addRow(self._btn_query)

        self._query_result = QLabel("—")
        self._query_result.setStyleSheet("font-weight: bold;")
        query_layout.addRow("Result:", self._query_result)

        query_group.setLayout(query_layout)
        layout.addWidget(query_group)

        # ── Pre-compute ──
        precompute_layout = QHBoxLayout()
        self._btn_slope_map = QPushButton("Compute Slope Map")
        self._btn_slope_map.clicked.connect(self._compute_slope_map)
        precompute_layout.addWidget(self._btn_slope_map)

        self._btn_contours = QPushButton("Generate Contours")
        self._btn_contours.clicked.connect(self._generate_contours)
        precompute_layout.addWidget(self._btn_contours)

        precompute_layout.addStretch()
        layout.addLayout(precompute_layout)

        layout.addStretch()

    # -- Slots ----------------------------------------------------------------

    def _browse_csv(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select DEM CSV File", "",
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        if filepath:
            self._file_path.setText(filepath)

    def _get_delimiter(self) -> str:
        if self._delim_other.isChecked():
            return self._delim_text.text() or ";"
        return ","

    def _load_csv(self) -> None:
        filepath = self._file_path.text().strip()
        if not filepath:
            QMessageBox.warning(self, "No File", "Please select a CSV file first.")
            return

        try:
            self._dem = DEMData.from_csv(
                filepath,
                x_col=self._x_col.value(),
                y_col=self._y_col.value(),
                z_col=self._z_col.value(),
                delimiter=self._get_delimiter(),
                has_header=self._has_header.isChecked(),
            )
            self._update_info()
            self.dem_loaded.emit(self._dem)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Failed",
                f"Failed to load DEM:\n{e}",
            )

    def _update_info(self) -> None:
        if self._dem is None:
            return
        stats = self._dem.stats()
        updates = {
            "Name": stats["name"],
            "Grid Shape": stats["n_cells"],
            "Cell Size": f"{stats['cell_size_m']:.2f} m" if stats["cell_size_m"] else "non-uniform",
            "X Range": f"[{stats['bounds_x'][0]:.1f}, {stats['bounds_x'][1]:.1f}]",
            "Y Range": f"[{stats['bounds_y'][0]:.1f}, {stats['bounds_y'][1]:.1f}]",
            "Z Min": f"{stats['z_min_m']:.2f} m",
            "Z Max": f"{stats['z_max_m']:.2f} m",
            "Z Range": f"{stats['z_range_m']:.2f} m",
        }
        for key, val in updates.items():
            if key in self._info_labels:
                self._info_labels[key].setText(str(val))
            self._info_labels[key].setStyleSheet("")

    def _query_elevation(self) -> None:
        if self._dem is None:
            QMessageBox.warning(self, "No DEM", "Please load a DEM first.")
            return
        try:
            x = float(self._query_x.text())
            y = float(self._query_y.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Coordinates", "Enter valid numbers.")
            return

        elev = self._dem.elevation(x, y)
        slope = self._dem.slope(x, y)
        aspect = self._dem.aspect(x, y)

        if np.isnan(elev):
            self._query_result.setText("⚠ Point outside DEM bounds")
        else:
            self._query_result.setText(
                f"Elevation: {elev:.2f} m  |  "
                f"Slope: {self._dem.slope_degrees(x, y):.1f}°  |  "
                f"Aspect: {aspect:.0f}°"
            )

    def _compute_slope_map(self) -> None:
        if self._dem is None:
            QMessageBox.warning(self, "No DEM", "Please load a DEM first.")
            return
        slope_map = self._dem.slope_map()
        mean_slope = float(np.nanmean(slope_map))
        max_slope = float(np.nanmax(slope_map))
        QMessageBox.information(
            self, "Slope Map",
            f"Mean slope: {mean_slope:.3f} (rise/run)\n"
            f"  ≈ {np.degrees(np.arctan(mean_slope)):.1f}°\n"
            f"Max slope: {max_slope:.3f}\n"
            f"  ≈ {np.degrees(np.arctan(max_slope)):.1f}°\n\n"
            f"Grid size: {slope_map.shape}",
        )

    def _generate_contours(self) -> None:
        if self._dem is None:
            QMessageBox.warning(self, "No DEM", "Please load a DEM first.")
            return
        try:
            contours = self._dem.generate_contours(n_levels=8)
            msg = f"Generated {len(contours)} contour lines:\n"
            for c in contours[:10]:
                msg += f"  • {c.elevation:.1f} m — {len(c.points)} pts, "
                msg += f"{'closed' if c.is_closed else 'open'}\n"
            if len(contours) > 10:
                msg += f"  ... and {len(contours) - 10} more"
            QMessageBox.information(self, "Contours Generated", msg)
        except Exception as e:
            QMessageBox.warning(self, "Contour Error", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# LOS Preview Widget
# ══════════════════════════════════════════════════════════════════════════════

class LOSPreviewWidget(QWidget):
    """A simple 2D diagram showing source, obstacles, and receptor."""

    _MARGIN = 40
    _SOURCE_COLOR = QColor(255, 120, 0)      # orange
    _TARGET_COLOR = QColor(0, 120, 255)       # blue
    _OBSTACLE_COLOR = QColor(120, 120, 120)   # gray
    _RAY_CLEAR = QColor(0, 200, 0, 100)       # green (semi-transparent)
    _RAY_BLOCKED = QColor(255, 0, 0, 150)     # red (semi-transparent)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 350)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().verticalPolicy(),
        )

        self._source_xyz: tuple = (0.0, 0.0, 0.0)
        self._target_xyz: tuple = (50.0, 0.0, 1.5)
        self._obstacles: List[Obstacle] = []
        self._los_result: LOSResult | None = None
        self._all_coords: List[tuple] = [(0, 0), (50, 0)]

    def set_scene(
        self,
        source_xyz: tuple,
        target_xyz: tuple,
        obstacles: List[Obstacle],
        los_result: LOSResult | None = None,
    ) -> None:
        """Set the scene to render."""
        self._source_xyz = source_xyz
        self._target_xyz = target_xyz
        self._obstacles = list(obstacles)
        self._los_result = los_result

        # Collect all coordinates for viewport fitting
        coords = [(source_xyz[0], source_xyz[1]),
                  (target_xyz[0], target_xyz[1])]
        for obs in obstacles:
            coords.append((obs.x - obs.length / 2, obs.y - obs.width / 2))
            coords.append((obs.x + obs.length / 2, obs.y + obs.width / 2))
        self._all_coords = coords

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        m = self._MARGIN

        # Fill background
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        # Compute viewport transform
        if not self._all_coords:
            return

        xs = [c[0] for c in self._all_coords]
        ys = [c[1] for c in self._all_coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add padding
        pad = max(x_max - x_min, y_max - y_min, 10.0) * 0.15
        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad

        # Aspect-ratio-preserving transform
        plot_w = w - 2 * m
        plot_h = h - 2 * m
        scale_x = plot_w / max(x_max - x_min, 1e-9)
        scale_y = plot_h / max(y_max - y_min, 1e-9)
        scale = min(scale_x, scale_y)

        # Center
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        def to_screen(px, py):
            sx = m + (px - cx) * scale + plot_w / 2.0
            sy = m + (cy - py) * scale + plot_h / 2.0
            return (int(sx), int(sy))

        # ── Draw obstacles ──
        for obs in self._obstacles:
            corners = obs.corners_2d()
            pts = [to_screen(c[0], c[1]) for c in corners]

            painter.setBrush(QBrush(self._OBSTACLE_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            from PyQt6.QtCore import QPoint
            polygon = [QPoint(p[0], p[1]) for p in pts]
            from PyQt6.QtGui import QPolygon
            painter.drawPolygon(QPolygon(polygon))

            # Label
            cx_obs, cy_obs = to_screen(obs.x, obs.y)
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Sans", 8))
            painter.drawText(cx_obs - 20, cy_obs - 8, obs.name[:15])

        # ── Draw source-target ray ──
        s_screen = to_screen(self._source_xyz[0], self._source_xyz[1])
        t_screen = to_screen(self._target_xyz[0], self._target_xyz[1])

        if self._los_result and self._los_result.is_blocked:
            pen = QPen(self._RAY_BLOCKED, 2, Qt.PenStyle.DashLine)
        elif self._los_result and self._los_result.status == LOSStatus.PARTIAL:
            pen = QPen(QColor(255, 200, 0, 150), 2, Qt.PenStyle.DashLine)
        else:
            pen = QPen(self._RAY_CLEAR, 2)

        painter.setPen(pen)
        painter.drawLine(s_screen[0], s_screen[1], t_screen[0], t_screen[1])

        # ── Draw source marker ──
        painter.setBrush(QBrush(self._SOURCE_COLOR))
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawEllipse(s_screen[0] - 6, s_screen[1] - 6, 12, 12)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        painter.drawText(s_screen[0] + 8, s_screen[1] + 4, "Source")

        # ── Draw target marker ──
        painter.setBrush(QBrush(self._TARGET_COLOR))
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawEllipse(t_screen[0] - 5, t_screen[1] - 5, 10, 10)
        painter.drawText(t_screen[0] + 8, t_screen[1] + 4, "Receptor")

        # ── Status text ──
        if self._los_result:
            status_text = {
                LOSStatus.CLEAR: "✅ LOS CLEAR",
                LOSStatus.PARTIAL: "⚠️ LOS PARTIALLY BLOCKED",
                LOSStatus.BLOCKED: "🚫 LOS BLOCKED",
                LOSStatus.SINGLE_BLOCKED: "🚫 SINGLE RAY BLOCKED",
            }.get(self._los_result.status, str(self._los_result.status))

            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Sans", 10, QFont.Weight.Bold))
            painter.drawText(10, h - 10, status_text)

        painter.end()


# ══════════════════════════════════════════════════════════════════════════════
# LOS Tab
# ══════════════════════════════════════════════════════════════════════════════

class LOSTab(QWidget):
    """Tab for configuring and running line-of-sight analyses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._los_engine = LOSEngine()
        self._obstacle_collection: ObstacleCollection | None = None
        self._setup_ui()

    def set_obstacle_collection(self, col: ObstacleCollection) -> None:
        """Update the obstacles fed to the LOS engine."""
        self._obstacle_collection = col
        self._los_engine.obstacles = list(col) if col else []
        self._btn_run.setEnabled(bool(col) and len(col) > 0)
        self._preview.set_scene(
            (0, 0, 0), (50, 0, 1.5),
            list(col) if col else [],
        )

    # -- UI setup -------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Left: controls, Right: preview ──
        hsplit = QHBoxLayout()

        # Left panel — controls
        controls = QVBoxLayout()

        # Source
        src_group = QGroupBox("Source")
        src_form = QFormLayout()
        src_coord = QHBoxLayout()
        self._src_x = QLineEdit("0.0")
        self._src_y = QLineEdit("0.0")
        self._src_z = QLineEdit("0.0")
        for label, w in [("X [m]:", self._src_x), ("Y [m]:", self._src_y),
                          ("Z [m]:", self._src_z)]:
            w.setMaximumWidth(80)
            w.setValidator(QDoubleValidator(-99999, 99999, 2))
            src_coord.addWidget(QLabel(label))
            src_coord.addWidget(w)
        src_coord.addStretch()
        src_form.addRow("Position:", src_coord)

        self._src_geometry = QComboBox()
        self._src_geometry.addItem("Pool Fire (Vertical Cylinder)", SourceGeometry.VERTICAL_CYLINDER.value)
        self._src_geometry.addItem("Jet Fire (Tilted Cylinder)", SourceGeometry.TILTED_CYLINDER.value)
        self._src_geometry.addItem("BLEVE (Sphere)", SourceGeometry.SPHERE.value)
        src_form.addRow("Geometry:", self._src_geometry)

        dim_layout = QHBoxLayout()
        self._src_d1 = QLineEdit("1.0")
        self._src_d1.setMaximumWidth(60)
        self._src_d1.setValidator(QDoubleValidator(0.01, 9999, 2))
        dim_layout.addWidget(QLabel("Dia [m]:"))
        dim_layout.addWidget(self._src_d1)
        self._src_d3 = QLineEdit("5.0")
        self._src_d3.setMaximumWidth(60)
        self._src_d3.setValidator(QDoubleValidator(0.01, 9999, 2))
        dim_layout.addWidget(QLabel("H/L [m]:"))
        dim_layout.addWidget(self._src_d3)
        dim_layout.addStretch()
        src_form.addRow("Dimensions:", dim_layout)
        src_group.setLayout(src_form)
        controls.addWidget(src_group)

        # Target
        tgt_group = QGroupBox("Receptor")
        tgt_form = QFormLayout()
        tgt_coord = QHBoxLayout()
        self._tgt_x = QLineEdit("50.0")
        self._tgt_y = QLineEdit("0.0")
        self._tgt_z = QLineEdit("1.5")
        for label, w in [("X [m]:", self._tgt_x), ("Y [m]:", self._tgt_y),
                          ("Z [m]:", self._tgt_z)]:
            w.setMaximumWidth(80)
            w.setValidator(QDoubleValidator(-99999, 99999, 2))
            tgt_coord.addWidget(QLabel(label))
            tgt_coord.addWidget(w)
        tgt_coord.addStretch()
        tgt_form.addRow("Position:", tgt_coord)
        tgt_group.setLayout(tgt_form)
        controls.addWidget(tgt_group)

        # Run
        self._btn_run = QPushButton("▶ Run LOS Analysis")
        self._btn_run.clicked.connect(self._run_los)
        self._btn_run.setEnabled(False)
        controls.addWidget(self._btn_run)

        # Results
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        controls.addWidget(self._results_label)

        controls.addStretch()
        hsplit.addLayout(controls, 1)

        # Right panel — preview
        self._preview = LOSPreviewWidget()
        hsplit.addWidget(self._preview, 3)

        layout.addLayout(hsplit)

    # -- Run LOS --------------------------------------------------------------

    def _run_los(self) -> None:
        if self._obstacle_collection is None or len(self._obstacle_collection) == 0:
            QMessageBox.warning(self, "No Obstacles",
                                "Add obstacles in the Obstacles tab first.")
            return

        try:
            sx = float(self._src_x.text())
            sy = float(self._src_y.text())
            sz = float(self._src_z.text())
            tx = float(self._tgt_x.text())
            ty = float(self._tgt_y.text())
            tz = float(self._tgt_z.text())
            d1 = float(self._src_d1.text())
            d3 = float(self._src_d3.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Enter valid numbers.")
            return

        geo_str = self._src_geometry.currentData()
        try:
            geometry = SourceGeometry(geo_str)
        except ValueError:
            geometry = SourceGeometry.VERTICAL_CYLINDER

        source_dims: tuple = (d1, d1, d3)

        self._los_engine.obstacles = list(self._obstacle_collection)

        result = self._los_engine.check_los(
            (sx, sy, sz), (tx, ty, tz),
            source_geometry=geometry,
            source_dims=source_dims,
            num_samples=24,
        )

        # Update preview
        self._preview.set_scene(
            (sx, sy, sz), (tx, ty, tz),
            list(self._obstacle_collection),
            result,
        )

        # Update results text
        lines = []
        lines.append(f"Status: {result.status.value}")
        lines.append(f"Blocked fraction: {result.blocked_fraction:.1%}")
        if result.blocking_obstacles:
            lines.append(f"Blocked by: {len(result.blocking_obstacles)} obstacle(s)")
            for oid in result.blocking_obstacles:
                obs = self._obstacle_collection.get(oid)
                if obs:
                    lines.append(f"  • {obs.name} ({obs.label})")
        else:
            lines.append("No obstacles in the ray path.")

        self._results_label.setText("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# Terrain Dialog
# ══════════════════════════════════════════════════════════════════════════════

class TerrainDialog(QDialog):
    """Main terrain configuration dialog with three tabs.

    Integrates obstacle management, DEM loading, and LOS analysis
    in a single comprehensive dialog.
    """

    terrain_data_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terrain & Obstacle Configuration")
        self.resize(900, 650)
        self.setMinimumSize(700, 500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Tab widget ──
        self._tabs = QTabWidget()

        # Tab 1: Obstacles
        self._obstacle_tab = ObstacleTab()
        self._tabs.addTab(self._obstacle_tab, "🏗️ Obstacles")

        # Tab 2: DEM
        self._dem_tab = DEMTab()
        self._tabs.addTab(self._dem_tab, "⛰️ DEM")

        # Tab 3: LOS Preview
        self._los_tab = LOSTab()
        self._tabs.addTab(self._los_tab, "📡 LOS Preview")

        # Wire: obstacles changed → update LOS tab
        self._obstacle_tab.obstacles_changed.connect(self._on_obstacles_changed)
        # Wire: DEM loaded → signal
        self._dem_tab.dem_loaded.connect(self._on_dem_loaded)
        # Wire: tab changed → update LOS preview if switching to LOS tab
        self._tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self._tabs)

        # ── Button box ──
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # -- Slots ----------------------------------------------------------------

    def _on_obstacles_changed(self) -> None:
        """Update LOS tab with current obstacles."""
        self._los_tab.set_obstacle_collection(self._obstacle_tab.collection)
        self.terrain_data_changed.emit()

    def _on_dem_loaded(self, dem: DEMData) -> None:
        """Notify about DEM load."""
        self.terrain_data_changed.emit()

    def _on_tab_changed(self, index: int) -> None:
        """When switching to LOS tab, refresh with current obstacles."""
        if index == 2:  # LOS tab
            self._los_tab.set_obstacle_collection(self._obstacle_tab.collection)

    # -- Public API -----------------------------------------------------------

    def set_obstacle_collection(self, col: ObstacleCollection) -> None:
        """Set the obstacle collection from outside."""
        self._obstacle_tab.set_collection(col)
        self._los_tab.set_obstacle_collection(col)

    def get_obstacle_collection(self) -> ObstacleCollection:
        """Get the current obstacle collection."""
        return self._obstacle_tab.collection

    def get_dem(self) -> DEMData | None:
        """Get the loaded DEM, if any."""
        return self._dem_tab.dem
