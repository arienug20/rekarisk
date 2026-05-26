"""
Rekarisk UI — Weather Input Dialog.

Provides a comprehensive dialog for configuring meteorological inputs
for dispersion calculations. Features three tabs:
    1. Single Weather — Direct wind/stability entry with preview
    2. Wind Rose    — Tabular wind rose frequency editor
    3. Weather File — CSV import with statistics
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..meteorology.stability import (
    StabilityClass,
    TerrainType,
    classify_stability,
    classify_stability_from_cloud,
    classify_stability_from_radiation,
    get_stability_description,
    list_terrain_types,
    mixing_height as pg_mixing_height,
    power_law_exponent,
    sigma_y,
    sigma_z,
    surface_roughness,
)
from ..meteorology.meteorology import (
    MeteorologicalState,
    atmospheric_density,
    wind_power_law,
)
from ..meteorology.wind_rose import (
    DEFAULT_SPEED_BINS,
    DEFAULT_SPEED_LABELS,
    DIRECTION_NAMES,
    N_DIRECTIONS,
    WindRoseData,
)
from ..meteorology.weather_data import WeatherDataset, WeatherObservation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STABILITY_CLASSES = ["A", "B", "C", "D", "E", "F"]
RADIATION_LEVELS = ["strong", "moderate", "slight"]
CLOUD_COVER_OPTIONS = {
    "Clear (0-3 oktas)": 2,
    "Partly cloudy (4-6 oktas)": 5,
    "Overcast (7-8 oktas)": 7,
}

# Default values
DEFAULT_WIND_SPEED = 3.0
DEFAULT_WIND_DIRECTION = 180.0
DEFAULT_TEMPERATURE_C = 25.0
DEFAULT_PRESSURE_KPA = 101.325
DEFAULT_HUMIDITY = 50.0
DEFAULT_Z0 = 0.1


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------


class _SectionLabel(QLabel):
    """Bold section label."""
    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        font = self.font()
        font.setBold(True)
        self.setFont(font)


# ---------------------------------------------------------------------------
# Tab 1: Single Weather
# ---------------------------------------------------------------------------


class SingleWeatherTab(QWidget):
    """Widget for entering a single meteorological state."""

    state_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stability_class: StabilityClass = "D"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Wind parameters ---
        wind_group = QGroupBox("Wind Parameters")
        wind_form = QFormLayout()

        self.wind_speed_edit = QLineEdit(str(DEFAULT_WIND_SPEED))
        self.wind_speed_edit.setValidator(QDoubleValidator(0.0, 50.0, 2))
        self.wind_speed_edit.textChanged.connect(self._on_change)
        wind_form.addRow("Wind Speed [m/s]:", self.wind_speed_edit)

        self.wind_dir_edit = QLineEdit(str(DEFAULT_WIND_DIRECTION))
        self.wind_dir_edit.setValidator(QDoubleValidator(0.0, 360.0, 1))
        self.wind_dir_edit.textChanged.connect(self._on_change)
        wind_form.addRow("Wind Direction [°]:", self.wind_dir_edit)

        self.ref_height_spin = QSpinBox()
        self.ref_height_spin.setRange(1, 100)
        self.ref_height_spin.setValue(10)
        self.ref_height_spin.setSuffix(" m")
        self.ref_height_spin.valueChanged.connect(self._on_change)
        wind_form.addRow("Reference Height:", self.ref_height_spin)

        wind_group.setLayout(wind_form)
        layout.addWidget(wind_group)

        # --- Atmospheric parameters ---
        atmos_group = QGroupBox("Atmospheric Conditions")
        atmos_form = QFormLayout()

        self.temp_edit = QLineEdit(str(DEFAULT_TEMPERATURE_C))
        self.temp_edit.setValidator(QDoubleValidator(-40.0, 60.0, 1))
        self.temp_edit.textChanged.connect(self._on_change)
        atmos_form.addRow("Temperature [°C]:", self.temp_edit)

        self.pressure_edit = QLineEdit(str(DEFAULT_PRESSURE_KPA))
        self.pressure_edit.setValidator(QDoubleValidator(80.0, 110.0, 3))
        self.pressure_edit.textChanged.connect(self._on_change)
        atmos_form.addRow("Pressure [kPa]:", self.pressure_edit)

        self.humidity_edit = QLineEdit(str(DEFAULT_HUMIDITY))
        self.humidity_edit.setValidator(QDoubleValidator(0.0, 100.0, 1))
        self.humidity_edit.textChanged.connect(self._on_change)
        atmos_form.addRow("Rel. Humidity [%]:", self.humidity_edit)

        atmos_group.setLayout(atmos_form)
        layout.addWidget(atmos_group)

        # --- Stability ---
        stab_group = QGroupBox("Stability Classification")
        stab_layout = QVBoxLayout()

        # Day/Night toggle
        daytime_layout = QHBoxLayout()
        self.daytime_check = QCheckBox("Daytime")
        self.daytime_check.setChecked(True)
        self.daytime_check.toggled.connect(self._on_daytime_toggled)
        daytime_layout.addWidget(self.daytime_check)
        daytime_layout.addStretch()
        stab_layout.addLayout(daytime_layout)

        # Solar radiation (daytime)
        self.radiation_widget = QWidget()
        rad_layout = QFormLayout(self.radiation_widget)
        self.radiation_combo = QComboBox()
        self.radiation_combo.addItems(RADIATION_LEVELS)
        self.radiation_combo.currentTextChanged.connect(self._on_stability_input_change)
        rad_layout.addRow("Solar Radiation:", self.radiation_combo)
        stab_layout.addWidget(self.radiation_widget)

        # Cloud cover (nighttime)
        self.cloud_widget = QWidget()
        cloud_layout = QFormLayout(self.cloud_widget)
        self.cloud_combo = QComboBox()
        self.cloud_combo.addItems(CLOUD_COVER_OPTIONS.keys())
        self.cloud_combo.currentTextChanged.connect(self._on_stability_input_change)
        cloud_layout.addRow("Cloud Cover:", self.cloud_combo)
        self.cloud_widget.setVisible(False)
        stab_layout.addWidget(self.cloud_widget)

        # Computed stability display
        stab_display_layout = QHBoxLayout()
        stab_display_layout.addWidget(QLabel("Stability Class:"))
        self.stability_label = QLabel("D — Neutral")
        font = self.stability_label.font()
        font.setBold(True)
        self.stability_label.setFont(font)
        stab_display_layout.addWidget(self.stability_label)
        stab_display_layout.addStretch()
        stab_layout.addLayout(stab_display_layout)

        # Manual override
        self.manual_stab_check = QCheckBox("Manual Override")
        self.manual_stab_check.toggled.connect(self._on_manual_stab_toggled)
        stab_layout.addWidget(self.manual_stab_check)

        self.manual_stab_combo = QComboBox()
        self.manual_stab_combo.addItems(STABILITY_CLASSES)
        self.manual_stab_combo.currentTextChanged.connect(self._on_manual_stab_change)
        self.manual_stab_combo.setVisible(False)
        stab_layout.addWidget(self.manual_stab_combo)

        stab_group.setLayout(stab_layout)
        layout.addWidget(stab_group)

        # --- Terrain ---
        terrain_group = QGroupBox("Terrain")
        terrain_form = QFormLayout()

        self.terrain_combo = QComboBox()
        terrain_types = list_terrain_types()
        self.terrain_combo.addItems(terrain_types)
        self.terrain_combo.setCurrentText("agricultural")
        self.terrain_combo.currentTextChanged.connect(self._on_change)
        terrain_form.addRow("Surface Roughness:", self.terrain_combo)

        terrain_group.setLayout(terrain_form)
        layout.addWidget(terrain_group)

        # --- Preview ---
        preview_group = QGroupBox("Preview at 100 m")
        preview_form = QFormLayout()

        self.preview_sigma_y = QLabel("—")
        self.preview_sigma_z = QLabel("—")
        self.preview_mixing_height = QLabel("—")
        self.preview_wind_50m = QLabel("—")
        self.preview_density = QLabel("—")

        preview_form.addRow("σ_y (100 m):", self.preview_sigma_y)
        preview_form.addRow("σ_z (100 m):", self.preview_sigma_z)
        preview_form.addRow("Mixing Height:", self.preview_mixing_height)
        preview_form.addRow("Wind at 50 m:", self.preview_wind_50m)
        preview_form.addRow("Air Density:", self.preview_density)

        preview_group.setLayout(preview_form)
        layout.addWidget(preview_group)

        layout.addStretch()

    def _on_daytime_toggled(self, checked: bool) -> None:
        self.radiation_widget.setVisible(checked)
        self.cloud_widget.setVisible(not checked)
        self._update_stability()
        self._on_change()

    def _on_stability_input_change(self) -> None:
        if not self.manual_stab_check.isChecked():
            self._update_stability()
        self._on_change()

    def _on_manual_stab_toggled(self, checked: bool) -> None:
        self.manual_stab_combo.setVisible(checked)
        if checked:
            self._stability_class = self.manual_stab_combo.currentText()  # type: ignore[assignment]
            self._update_preview()
            self.state_changed.emit()
        else:
            self._update_stability()

    def _on_manual_stab_change(self, text: str) -> None:
        self._stability_class = text  # type: ignore[assignment]
        self._update_preview()
        self.state_changed.emit()

    def _on_change(self) -> None:
        if not self.manual_stab_check.isChecked():
            self._update_stability()
        self._update_preview()
        self.state_changed.emit()

    def _update_stability(self) -> None:
        """Recompute stability class from current inputs."""
        try:
            ws = float(self.wind_speed_edit.text())
        except ValueError:
            ws = DEFAULT_WIND_SPEED

        if self.daytime_check.isChecked():
            rad_level = self.radiation_combo.currentText()
            sc = classify_stability_from_radiation(ws, rad_level)
        else:
            cloud_key = self.cloud_combo.currentText()
            c_oktas = CLOUD_COVER_OPTIONS.get(cloud_key, 2)
            sc = classify_stability_from_cloud(ws, c_oktas)

        self._stability_class = sc
        self.stability_label.setText(
            f"{sc} — {get_stability_description(sc)}"
        )

    def _update_preview(self) -> None:
        """Update preview panel."""
        try:
            ws = float(self.wind_speed_edit.text())
        except ValueError:
            ws = DEFAULT_WIND_SPEED
        try:
            tc = float(self.temp_edit.text())
        except ValueError:
            tc = DEFAULT_TEMPERATURE_C
        try:
            p_kpa = float(self.pressure_edit.text())
        except ValueError:
            p_kpa = DEFAULT_PRESSURE_KPA
        try:
            rh = float(self.humidity_edit.text())
        except ValueError:
            rh = DEFAULT_HUMIDITY

        sc = self._stability_class
        t_k = tc + 273.15
        p_pa = p_kpa * 1000.0

        # Sigma values at 100 m
        sy = sigma_y(100.0, sc, "rural")
        sz = sigma_z(100.0, sc, "rural")
        self.preview_sigma_y.setText(f"{sy:.1f} m")
        self.preview_sigma_z.setText(f"{sz:.1f} m")

        # Mixing height
        mh = pg_mixing_height(sc, self.daytime_check.isChecked())
        self.preview_mixing_height.setText(f"{mh:.0f} m")

        # Wind at 50 m
        u50 = wind_power_law(50.0, ws, self.ref_height_spin.value(), sc)
        self.preview_wind_50m.setText(f"{u50:.2f} m/s")

        # Air density
        rho = atmospheric_density(t_k, p_pa, rh)
        self.preview_density.setText(f"{rho:.4f} kg/m³")

    def get_state(self) -> MeteorologicalState:
        """Build MeteorologicalState from current inputs."""
        try:
            ws = float(self.wind_speed_edit.text())
        except ValueError:
            ws = DEFAULT_WIND_SPEED
        try:
            wd = float(self.wind_dir_edit.text())
        except ValueError:
            wd = DEFAULT_WIND_DIRECTION
        try:
            tc = float(self.temp_edit.text())
        except ValueError:
            tc = DEFAULT_TEMPERATURE_C
        try:
            p_kpa = float(self.pressure_edit.text())
        except ValueError:
            p_kpa = DEFAULT_PRESSURE_KPA
        try:
            rh = float(self.humidity_edit.text())
        except ValueError:
            rh = DEFAULT_HUMIDITY

        terrain_name = self.terrain_combo.currentText()
        z0 = surface_roughness(terrain_name)

        return MeteorologicalState(
            wind_speed_ms=ws,
            wind_direction_deg=wd,
            reference_height_m=self.ref_height_spin.value(),
            ambient_temperature_k=tc + 273.15,
            ambient_pressure_pa=p_kpa * 1000.0,
            relative_humidity_pct=rh,
            cloud_cover_oktas=(
                CLOUD_COVER_OPTIONS.get(self.cloud_combo.currentText(), 4)
            ),
            is_daytime=self.daytime_check.isChecked(),
            surface_roughness_m=z0,
            stability_class=self._stability_class,
        )

    def set_state(self, state: MeteorologicalState) -> None:
        """Load a MeteorologicalState into the UI."""
        self.wind_speed_edit.setText(f"{state.wind_speed_ms:.1f}")
        self.wind_dir_edit.setText(f"{state.wind_direction_deg:.0f}")
        self.ref_height_spin.setValue(int(state.reference_height_m))
        self.temp_edit.setText(f"{state.ambient_temperature_k - 273.15:.1f}")
        self.pressure_edit.setText(f"{state.ambient_pressure_pa / 1000.0:.3f}")
        self.humidity_edit.setText(f"{state.relative_humidity_pct:.0f}")
        self.daytime_check.setChecked(state.is_daytime)

        if state.stability_class:
            self.manual_stab_check.setChecked(True)
            self.manual_stab_combo.setCurrentText(state.stability_class)
            self._stability_class = state.stability_class
        else:
            self.manual_stab_check.setChecked(False)
            self._update_stability()

        self._update_preview()


# ---------------------------------------------------------------------------
# Tab 2: Wind Rose Editor
# ---------------------------------------------------------------------------


class WindRoseTab(QWidget):
    """Widget for editing wind rose frequency data."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._wind_rose = WindRoseData()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(self.clear_btn)

        self.normalize_btn = QPushButton("Normalize to 1.0")
        self.normalize_btn.clicked.connect(self._normalize)
        toolbar.addWidget(self.normalize_btn)

        self.import_csv_btn = QPushButton("Import CSV...")
        self.import_csv_btn.clicked.connect(self._import_csv)
        toolbar.addWidget(self.import_csv_btn)

        self.export_csv_btn = QPushButton("Export CSV...")
        self.export_csv_btn.clicked.connect(self._export_csv)
        toolbar.addWidget(self.export_csv_btn)

        toolbar.addStretch()

        calm_label = QLabel("Calm Count:")
        self.calm_edit = QLineEdit("0")
        self.calm_edit.setFixedWidth(80)
        self.calm_edit.setValidator(QDoubleValidator(0.0, 1e9, 1))
        toolbar.addWidget(calm_label)
        toolbar.addWidget(self.calm_edit)

        layout.addLayout(toolbar)

        # Table: rows = speed classes, columns = directions
        self.table = QTableWidget()
        self.table.setRowCount(len(DEFAULT_SPEED_BINS))
        self.table.setColumnCount(N_DIRECTIONS)

        # Column headers = direction names
        self.table.setHorizontalHeaderLabels(DIRECTION_NAMES)

        # Row headers = speed labels
        self.table.setVerticalHeaderLabels(DEFAULT_SPEED_LABELS)

        # Set reasonable column widths
        for i in range(N_DIRECTIONS):
            self.table.setColumnWidth(i, 55)

        # Connect cell change
        self.table.cellChanged.connect(self._on_cell_changed)

        layout.addWidget(self.table)

        # Summary panel
        summary_group = QGroupBox("Wind Rose Summary")
        summary_layout = QFormLayout()

        self.summary_dominant = QLabel("—")
        self.summary_calm = QLabel("—")
        self.summary_mean_ws = QLabel("—")
        summary_layout.addRow("Dominant Direction:", self.summary_dominant)
        summary_layout.addRow("Calm Fraction:", self.summary_calm)
        summary_layout.addRow("Mean Wind Speed:", self.summary_mean_ws)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Initialize with empty data
        self._refresh_table_from_data()

    def _on_cell_changed(self, row: int, col: int) -> None:
        """Handle user editing a cell."""
        item = self.table.item(row, col)
        if item:
            try:
                val = float(item.text())
                item.setText(f"{val:.4f}")
            except ValueError:
                item.setText("0.0000")

    def _refresh_table_from_data(self) -> None:
        """Update table with current wind rose data."""
        self.table.blockSignals(True)

        probs = self._wind_rose.joint_probability_distribution()
        for i in range(len(DEFAULT_SPEED_BINS)):
            for j in range(N_DIRECTIONS):
                val = probs[i, j] if i < probs.shape[0] and j < probs.shape[1] else 0.0
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j, item)

        self.table.blockSignals(False)

        # Update summary
        calm = self._wind_rose.calm_fraction()
        dom_idx = self._wind_rose.dominant_direction()
        dom_name = DIRECTION_NAMES[dom_idx % N_DIRECTIONS]

        self.summary_dominant.setText(f"{dom_name} ({dom_idx * 22.5:.0f}°)")
        self.summary_calm.setText(f"{calm:.4f} ({self._wind_rose.calm_count:.0f} obs)")
        self.summary_mean_ws.setText(f"{self._wind_rose.mean_wind_speed():.2f} m/s")
        self.calm_edit.setText(f"{self._wind_rose.calm_count:.0f}")

    def _collect_table_data(self) -> None:
        """Read table data into wind rose object."""
        prob_matrix = []
        total_sum = 0.0

        for i in range(len(DEFAULT_SPEED_BINS)):
            row = []
            for j in range(N_DIRECTIONS):
                item = self.table.item(i, j)
                if item:
                    try:
                        val = float(item.text())
                    except ValueError:
                        val = 0.0
                else:
                    val = 0.0
                row.append(max(0.0, val))
                total_sum += row[-1]
            prob_matrix.append(row)

        prob_arr = __import__("numpy").array(prob_matrix, dtype=float)

        # Handle calm
        try:
            calm_val = float(self.calm_edit.text())
        except ValueError:
            calm_val = 0.0

        if total_sum + calm_val > 0:
            # Normalize
            normalizer = total_sum + calm_val
            prob_arr /= normalizer
            calm_prob = calm_val / normalizer
        else:
            calm_prob = 0.0

        self._wind_rose.set_probabilities(prob_arr, calm_prob)

    def _clear(self) -> None:
        """Clear all data."""
        self._wind_rose = WindRoseData()
        self._refresh_table_from_data()

    def _normalize(self) -> None:
        """Normalize table values to sum to 1."""
        self._collect_table_data()
        probs = self._wind_rose.joint_probability_distribution()
        calm = self._wind_rose.calm_fraction()
        total = probs.sum() + calm

        if total > 0:
            self._wind_rose.set_probabilities(probs / total, calm / total)
        else:
            # Set equal probabilities
            n_cells = N_DIRECTIONS * len(DEFAULT_SPEED_BINS)
            probs = __import__("numpy").ones((len(DEFAULT_SPEED_BINS), N_DIRECTIONS)) / n_cells
            self._wind_rose.set_probabilities(probs, 0.0)

        self._refresh_table_from_data()

    def _import_csv(self) -> None:
        """Import wind rose data from CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Wind Rose CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path) as f:
                text = f.read()
            self._wind_rose = WindRoseData()
            self._wind_rose.from_csv(text)
            self._refresh_table_from_data()
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _export_csv(self) -> None:
        """Export wind rose data to CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Wind Rose CSV",
            "wind_rose.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            self._collect_table_data()
            csv_text = self._wind_rose.to_csv()
            with open(path, "w") as f:
                f.write(csv_text)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def get_wind_rose(self) -> WindRoseData:
        """Get the current wind rose data."""
        self._collect_table_data()
        return self._wind_rose

    def set_wind_rose(self, wr: WindRoseData) -> None:
        """Set wind rose data."""
        self._wind_rose = wr
        self._refresh_table_from_data()


# ---------------------------------------------------------------------------
# Tab 3: Weather File
# ---------------------------------------------------------------------------


class WeatherFileTab(QWidget):
    """Widget for importing and analyzing weather data files."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._dataset: Optional[WeatherDataset] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file loaded...")
        file_layout.addWidget(self.file_path_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        file_layout.addWidget(self.browse_btn)

        layout.addLayout(file_layout)

        # Options
        options_layout = QHBoxLayout()

        self.has_header_check = QCheckBox("Has Header")
        self.has_header_check.setChecked(True)
        options_layout.addWidget(self.has_header_check)

        options_layout.addStretch()

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self._load_file)
        self.load_btn.setEnabled(False)
        options_layout.addWidget(self.load_btn)

        layout.addLayout(options_layout)

        # Preview text
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout()
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()

        self.stat_count = QLabel("—")
        self.stat_date_range = QLabel("—")
        self.stat_mean_ws = QLabel("—")
        self.stat_mean_temp = QLabel("—")
        self.stat_dom_stability = QLabel("—")
        self.stat_calm = QLabel("—")

        stats_layout.addRow("Observations:", self.stat_count)
        stats_layout.addRow("Date Range:", self.stat_date_range)
        stats_layout.addRow("Mean Wind Speed:", self.stat_mean_ws)
        stats_layout.addRow("Mean Temperature:", self.stat_mean_temp)
        stats_layout.addRow("Dominant Stability:", self.stat_dom_stability)
        stats_layout.addRow("Calm Fraction:", self.stat_calm)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

    def _browse(self) -> None:
        """Browse for a weather data file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Weather Data File",
            "",
            "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.file_path_edit.setText(path)
            self.load_btn.setEnabled(True)
            # Show preview
            try:
                with open(path) as f:
                    preview = "".join(f.readlines()[:20])
                self.preview_text.setText(preview)
            except Exception:
                self.preview_text.setText("(Cannot read file)")

    def _load_file(self) -> None:
        """Load and analyze the selected weather data file."""
        path = self.file_path_edit.text()
        if not path:
            return

        try:
            with open(path) as f:
                content = f.read()

            if path.endswith(".json"):
                self._dataset = WeatherDataset.from_json(content)
            else:
                self._dataset = WeatherDataset.from_csv(
                    content,
                    has_header=self.has_header_check.isChecked(),
                )

            self._update_statistics()
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _update_statistics(self) -> None:
        """Update the statistics display."""
        if not self._dataset or len(self._dataset) == 0:
            return

        ds = self._dataset
        self.stat_count.setText(f"{len(ds):,}")

        # Date range
        timestamps = [obs.timestamp for obs in ds.observations]
        if timestamps:
            t_min = min(timestamps)
            t_max = max(timestamps)
            self.stat_date_range.setText(f"{t_min:%Y-%m-%d %H:%M} to {t_max:%Y-%m-%d %H:%M}")

        # Wind
        ws = ds.mean_wind_speed()
        self.stat_mean_ws.setText(f"{ws:.2f} m/s")

        # Temperature
        tk = ds.mean_temperature()
        self.stat_mean_temp.setText(f"{tk:.1f} K ({tk - 273.15:.1f} °C)")

        # Stability
        dom_stab = ds.dominant_stability()
        if dom_stab:
            self.stat_dom_stability.setText(f"{dom_stab} — {get_stability_description(dom_stab)}")

        # Calm fraction
        wr = ds.to_wind_rose()
        self.stat_calm.setText(f"{wr.calm_fraction():.4f}")

    def get_dataset(self) -> Optional[WeatherDataset]:
        """Get the loaded dataset."""
        return self._dataset

    def get_wind_rose(self) -> Optional[WindRoseData]:
        """Get wind rose from loaded dataset."""
        if self._dataset:
            return self._dataset.to_wind_rose()
        return None


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------


class WeatherDialog(QDialog):
    """Dialog for configuring meteorological inputs.

    Three tabs:
        1. Single Weather — Direct entry with live preview
        2. Wind Rose — Frequency table editor
        3. Weather File — CSV import with statistics

    Can save/load presets as JSON files.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Weather Configuration")
        self.setMinimumSize(800, 600)
        self._preset_path: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()

        self.single_tab = SingleWeatherTab()
        self.wind_rose_tab = WindRoseTab()
        self.weather_file_tab = WeatherFileTab()

        self.tabs.addTab(self.single_tab, "Single Weather")
        self.tabs.addTab(self.wind_rose_tab, "Wind Rose")
        self.tabs.addTab(self.weather_file_tab, "Weather File")

        layout.addWidget(self.tabs)

        # Preset toolbar
        preset_layout = QHBoxLayout()

        self.save_preset_btn = QPushButton("Save Preset...")
        self.save_preset_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(self.save_preset_btn)

        self.load_preset_btn = QPushButton("Load Preset...")
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_layout.addWidget(self.load_preset_btn)

        preset_layout.addStretch()

        layout.addLayout(preset_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_meteorological_state(self) -> MeteorologicalState:
        """Get the meteorological state from the single weather tab."""
        return self.single_tab.get_state()

    def set_meteorological_state(self, state: MeteorologicalState) -> None:
        """Set the meteorological state in the single weather tab."""
        self.single_tab.set_state(state)
        self.tabs.setCurrentIndex(0)

    def get_wind_rose(self) -> WindRoseData:
        """Get wind rose data from the active tab."""
        current_idx = self.tabs.currentIndex()
        if current_idx == 1:
            return self.wind_rose_tab.get_wind_rose()
        elif current_idx == 2:
            wr = self.weather_file_tab.get_wind_rose()
            return wr if wr else WindRoseData()
        else:
            return WindRoseData()

    def _save_preset(self) -> None:
        """Save current configuration as a JSON preset."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Weather Preset",
            "weather_preset.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        data = {
            "version": "1.0",
            "type": "weather_preset",
            "meteorological_state": self.single_tab.get_state().to_dict(),
            "wind_rose": self.wind_rose_tab.get_wind_rose().to_dict(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._preset_path = path
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _load_preset(self) -> None:
        """Load a weather preset from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Weather Preset",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Load meteorological state
            state = MeteorologicalState.from_dict(data.get("meteorological_state", {}))
            self.single_tab.set_state(state)

            # Load wind rose
            wr_dict = data.get("wind_rose", {})
            if wr_dict:
                self.wind_rose_tab.set_wind_rose(WindRoseData.from_dict(wr_dict))

            self._preset_path = path
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    @staticmethod
    def get_weather(
        parent: Optional[QWidget] = None,
        initial_state: Optional[MeteorologicalState] = None,
    ) -> Optional[Dict]:
        """Convenience static method: show dialog and return weather config.

        Args:
            parent: Parent widget.
            initial_state: Initial MeteorologicalState to populate.

        Returns:
            Dict with 'meteorological_state' and 'wind_rose' keys, or None if cancelled.
        """
        dialog = WeatherDialog(parent)
        if initial_state:
            dialog.set_meteorological_state(initial_state)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {
                "meteorological_state": dialog.get_meteorological_state(),
                "wind_rose": dialog.get_wind_rose(),
            }
        return None
