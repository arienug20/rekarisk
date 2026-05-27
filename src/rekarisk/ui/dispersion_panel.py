"""
Rekarisk UI — Dispersion Input Panel.

Provides a PyQt6 tabbed input panel for dispersion calculations:
  - Source: Release parameters (rate, mass, duration)
  - Weather: Atmospheric conditions
  - Receptor Grid: Spatial grid definition
  - Advanced: Decay, deposition, averaging time

Connected to the DispersionDispatcher for auto model selection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QFormLayout, QGroupBox, QCheckBox,
    QSplitter, QFrame, QProgressBar, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal

from rekarisk.meteorology.stability import StabilityClass, list_terrain_types
from rekarisk.core.constants import P_ATM, T_0C

STABILITY_CLASSES = ["A", "B", "C", "D", "E", "F"]
RELEASE_TYPES = ["continuous", "instantaneous"]
TERRAIN_TYPES = list_terrain_types() if list_terrain_types() else ["rural", "urban"]


class DispersionPanel(QWidget):
    """Main dispersion input panel with tabbed interface.

    Tabs:
      - Source: Link to source term results or manual entry
      - Weather: Link to weather data or manual entry
      - Receptor Grid: Define evaluation grid
      - Advanced: Decay rate, deposition, sampling time

    Signals:
        calculation_requested: Emitted with full params dict when Run is clicked.
    """

    calculation_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create tabs
        self._source_tab = SourceTab()
        self._weather_tab = WeatherTab()
        self._grid_tab = ReceptorGridTab()
        self._advanced_tab = AdvancedTab()

        self.tabs.addTab(self._source_tab, "📦 Source")
        self.tabs.addTab(self._weather_tab, "🌤️ Weather")
        self.tabs.addTab(self._grid_tab, "📐 Grid")
        self.tabs.addTab(self._advanced_tab, "⚙️ Advanced")

        layout.addWidget(self.tabs)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Run button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.run_button = QPushButton("🚀 Run Dispersion")
        self.run_button.setMinimumHeight(36)
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.run_button.clicked.connect(self._on_run)
        btn_layout.addWidget(self.run_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_run(self):
        """Collect all parameters and emit calculation_requested."""
        params = self.get_all_params()
        # Validate
        errors = []
        if params.get("source_rate", 0) <= 0 and params.get("source_mass", 0) <= 0:
            errors.append("Source rate or mass must be > 0")
        if params.get("wind_speed", 0) <= 0:
            errors.append("Wind speed must be > 0")

        if errors:
            QMessageBox.warning(
                self, "Validation Error",
                "Please fix the following:\n• " + "\n• ".join(errors)
            )
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate
        self.run_button.setEnabled(False)

        self.calculation_requested.emit(params)

    def on_calculation_complete(self):
        """Reset UI after calculation completes."""
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)

    def get_all_params(self) -> Dict[str, Any]:
        """Collect all parameters from all tabs."""
        params = {}
        params.update(self._source_tab.get_params())
        params.update(self._weather_tab.get_params())
        params.update(self._grid_tab.get_params())
        params.update(self._advanced_tab.get_params())
        return params

    def set_substance(self, substance) -> None:
        """Pre-fill source tab from a Substance database entry."""
        is_gas = getattr(substance, 'is_gas_at_ambient', False)

        # Phase
        self._source_tab.phase_combo.blockSignals(True)
        self._source_tab.phase_combo.setCurrentText("gas" if is_gas else "liquid")
        self._source_tab.phase_combo.blockSignals(False)

        # Cloud density
        rho = substance.vapor_density if is_gas else substance.liquid_density
        if rho is not None:
            self._source_tab.cloud_density_spin.setValue(rho)

        # Molecular weight (widget expects g/mol — same as DB)
        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self._source_tab.mw_spin.setValue(mw)

    def set_source_params(self, **kwargs):
        """Set source parameters from external data (e.g., source term result)."""
        self._source_tab.set_params(**kwargs)

    def set_weather_params(self, **kwargs):
        """Set weather parameters from external data (e.g., weather dialog)."""
        self._weather_tab.set_params(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Source Tab
# ══════════════════════════════════════════════════════════════════════════════

class SourceTab(QWidget):
    """Source term input tab for dispersion calculations.

    Supports both continuous and instantaneous releases, with fields for
    mass rate, total mass, duration, cloud density, temperature, and
    release geometry.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Release type
        type_group = QGroupBox("Release Type")
        type_form = QFormLayout()

        self.release_type_combo = QComboBox()
        self.release_type_combo.addItems(RELEASE_TYPES)
        self.release_type_combo.currentTextChanged.connect(self._on_type_changed)
        type_form.addRow("Type:", self.release_type_combo)

        type_group.setLayout(type_form)
        layout.addWidget(type_group)

        # Release parameters
        param_group = QGroupBox("Release Parameters")
        param_form = QFormLayout()

        self.mass_rate_spin = QDoubleSpinBox()
        self.mass_rate_spin.setRange(0.0, 1e6)
        self.mass_rate_spin.setValue(1.0)
        self.mass_rate_spin.setDecimals(4)
        self.mass_rate_spin.setSuffix(" kg/s")
        param_form.addRow("Mass Rate (Q):", self.mass_rate_spin)

        self.mass_total_spin = QDoubleSpinBox()
        self.mass_total_spin.setRange(0.0, 1e9)
        self.mass_total_spin.setValue(0.0)
        self.mass_total_spin.setDecimals(2)
        self.mass_total_spin.setSuffix(" kg")
        self.mass_total_spin.setEnabled(False)
        param_form.addRow("Total Mass:", self.mass_total_spin)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 86400 * 365)
        self.duration_spin.setValue(0.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" s")
        param_form.addRow("Duration:", self.duration_spin)

        self.cloud_density_spin = QDoubleSpinBox()
        self.cloud_density_spin.setRange(0.1, 100.0)
        self.cloud_density_spin.setValue(1.2)
        self.cloud_density_spin.setDecimals(3)
        self.cloud_density_spin.setSuffix(" kg/m³")
        param_form.addRow("Cloud Density:", self.cloud_density_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(50.0, 3000.0)
        self.temp_spin.setValue(298.15)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setSuffix(" K")
        param_form.addRow("Temperature:", self.temp_spin)

        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setRange(1.0, 500.0)
        self.mw_spin.setValue(29.0)
        self.mw_spin.setDecimals(1)
        self.mw_spin.setSuffix(" g/mol")
        param_form.addRow("MW:", self.mw_spin)

        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["gas", "liquid", "two_phase"])
        param_form.addRow("Phase:", self.phase_combo)

        param_group.setLayout(param_form)
        layout.addWidget(param_group)

        # Release geometry
        geom_group = QGroupBox("Release Geometry")
        geom_form = QFormLayout()

        self.release_height_spin = QDoubleSpinBox()
        self.release_height_spin.setRange(0.0, 500.0)
        self.release_height_spin.setValue(0.0)
        self.release_height_spin.setDecimals(1)
        self.release_height_spin.setSuffix(" m")
        geom_form.addRow("Height:", self.release_height_spin)

        self.release_diameter_spin = QDoubleSpinBox()
        self.release_diameter_spin.setRange(0.0, 10.0)
        self.release_diameter_spin.setValue(0.0)
        self.release_diameter_spin.setDecimals(3)
        self.release_diameter_spin.setSuffix(" m")
        geom_form.addRow("Diameter:", self.release_diameter_spin)

        self.exit_velocity_spin = QDoubleSpinBox()
        self.exit_velocity_spin.setRange(0.0, 1000.0)
        self.exit_velocity_spin.setValue(0.0)
        self.exit_velocity_spin.setDecimals(1)
        self.exit_velocity_spin.setSuffix(" m/s")
        geom_form.addRow("Exit Velocity:", self.exit_velocity_spin)

        self.heat_rate_spin = QDoubleSpinBox()
        self.heat_rate_spin.setRange(0.0, 1e9)
        self.heat_rate_spin.setValue(0.0)
        self.heat_rate_spin.setDecimals(1)
        self.heat_rate_spin.setSuffix(" W")
        geom_form.addRow("Heat Rate:", self.heat_rate_spin)

        geom_group.setLayout(geom_form)
        layout.addWidget(geom_group)

        layout.addStretch()

    def _on_type_changed(self, text: str):
        """Enable/disable fields based on release type."""
        if text == "continuous":
            self.mass_rate_spin.setEnabled(True)
            self.mass_total_spin.setEnabled(False)
            self.duration_spin.setEnabled(True)
        else:
            self.mass_rate_spin.setEnabled(False)
            self.mass_total_spin.setEnabled(True)
            self.duration_spin.setEnabled(False)

    def get_params(self) -> Dict[str, Any]:
        return {
            "release_type": self.release_type_combo.currentText(),
            "source_rate": self.mass_rate_spin.value(),
            "source_mass": self.mass_total_spin.value(),
            "duration": self.duration_spin.value(),
            "cloud_density": self.cloud_density_spin.value(),
            "temperature": self.temp_spin.value(),
            "molecular_weight": self.mw_spin.value(),
            "phase": self.phase_combo.currentText(),
            "release_height": self.release_height_spin.value(),
            "release_diameter": self.release_diameter_spin.value(),
            "exit_velocity": self.exit_velocity_spin.value(),
            "heat_release_rate": self.heat_rate_spin.value(),
        }

    def set_params(self, **kwargs):
        if "release_type" in kwargs:
            idx = self.release_type_combo.findText(kwargs["release_type"])
            if idx >= 0:
                self.release_type_combo.setCurrentIndex(idx)
        if "source_rate" in kwargs:
            self.mass_rate_spin.setValue(kwargs["source_rate"])
        if "source_mass" in kwargs:
            self.mass_total_spin.setValue(kwargs["source_mass"])
        if "duration" in kwargs:
            self.duration_spin.setValue(kwargs["duration"])
        if "cloud_density" in kwargs:
            self.cloud_density_spin.setValue(kwargs["cloud_density"])
        if "temperature" in kwargs:
            self.temp_spin.setValue(kwargs["temperature"])
        if "molecular_weight" in kwargs:
            self.mw_spin.setValue(kwargs["molecular_weight"])
        if "phase" in kwargs:
            idx = self.phase_combo.findText(kwargs["phase"])
            if idx >= 0:
                self.phase_combo.setCurrentIndex(idx)
        if "release_height" in kwargs:
            self.release_height_spin.setValue(kwargs["release_height"])
        if "release_diameter" in kwargs:
            self.release_diameter_spin.setValue(kwargs["release_diameter"])
        if "exit_velocity" in kwargs:
            self.exit_velocity_spin.setValue(kwargs["exit_velocity"])
        if "heat_release_rate" in kwargs:
            self.heat_rate_spin.setValue(kwargs["heat_release_rate"])


# ══════════════════════════════════════════════════════════════════════════════
# Weather Tab
# ══════════════════════════════════════════════════════════════════════════════

class WeatherTab(QWidget):
    """Weather/meteorology input tab for dispersion calculations.

    Parameters can be manually entered or linked from weather_dialog.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Wind
        wind_group = QGroupBox("Wind Conditions")
        wind_form = QFormLayout()

        self.wind_speed_spin = QDoubleSpinBox()
        self.wind_speed_spin.setRange(0.5, 50.0)
        self.wind_speed_spin.setValue(5.0)
        self.wind_speed_spin.setDecimals(1)
        self.wind_speed_spin.setSuffix(" m/s")
        wind_form.addRow("Wind Speed:", self.wind_speed_spin)

        self.wind_dir_spin = QDoubleSpinBox()
        self.wind_dir_spin.setRange(0.0, 360.0)
        self.wind_dir_spin.setValue(0.0)
        self.wind_dir_spin.setDecimals(0)
        self.wind_dir_spin.setSuffix(" °")
        wind_form.addRow("Direction:", self.wind_dir_spin)

        self.ref_height_spin = QDoubleSpinBox()
        self.ref_height_spin.setRange(1.0, 200.0)
        self.ref_height_spin.setValue(10.0)
        self.ref_height_spin.setDecimals(1)
        self.ref_height_spin.setSuffix(" m")
        wind_form.addRow("Ref. Height:", self.ref_height_spin)

        wind_group.setLayout(wind_form)
        layout.addWidget(wind_group)

        # Stability
        stab_group = QGroupBox("Atmospheric Stability")
        stab_form = QFormLayout()

        self.stability_combo = QComboBox()
        self.stability_combo.addItems(STABILITY_CLASSES)
        self.stability_combo.setCurrentText("D")
        stab_form.addRow("Stability Class:", self.stability_combo)

        stability_desc = {
            "A": "Very unstable",
            "B": "Moderately unstable",
            "C": "Slightly unstable",
            "D": "Neutral",
            "E": "Slightly stable",
            "F": "Moderately stable",
        }

        self.stability_label = QLabel(stability_desc["D"])
        self.stability_label.setStyleSheet("color: #555555; font-style: italic;")
        stab_form.addRow("", self.stability_label)

        self.stability_combo.currentTextChanged.connect(
            lambda t: self.stability_label.setText(
                stability_desc.get(t, "")
            )
        )

        self.terrain_combo = QComboBox()
        for t in TERRAIN_TYPES:
            self.terrain_combo.addItem(t)
        idx = self.terrain_combo.findText("rural")
        if idx >= 0:
            self.terrain_combo.setCurrentIndex(idx)
        elif self.terrain_combo.count() > 0:
            self.terrain_combo.setCurrentIndex(0)
        stab_form.addRow("Terrain:", self.terrain_combo)

        stab_group.setLayout(stab_form)
        layout.addWidget(stab_group)

        # Ambient
        amb_group = QGroupBox("Ambient Conditions")
        amb_form = QFormLayout()

        self.amb_temp_spin = QDoubleSpinBox()
        self.amb_temp_spin.setRange(200.0, 350.0)
        self.amb_temp_spin.setValue(298.15)
        self.amb_temp_spin.setDecimals(2)
        self.amb_temp_spin.setSuffix(" K")
        amb_form.addRow("Temperature:", self.amb_temp_spin)

        self.amb_press_spin = QDoubleSpinBox()
        self.amb_press_spin.setRange(50000.0, 110000.0)
        self.amb_press_spin.setValue(P_ATM)
        self.amb_press_spin.setDecimals(0)
        self.amb_press_spin.setSuffix(" Pa")
        amb_form.addRow("Pressure:", self.amb_press_spin)

        self.humidity_spin = QDoubleSpinBox()
        self.humidity_spin.setRange(0.0, 100.0)
        self.humidity_spin.setValue(50.0)
        self.humidity_spin.setDecimals(1)
        self.humidity_spin.setSuffix(" %")
        amb_form.addRow("Humidity:", self.humidity_spin)

        self.cloud_spin = QDoubleSpinBox()
        self.cloud_spin.setRange(0.0, 8.0)
        self.cloud_spin.setValue(4.0)
        self.cloud_spin.setDecimals(0)
        self.cloud_spin.setSuffix(" oktas")
        amb_form.addRow("Cloud Cover:", self.cloud_spin)

        self.solar_spin = QDoubleSpinBox()
        self.solar_spin.setRange(0.0, 1200.0)
        self.solar_spin.setValue(500.0)
        self.solar_spin.setDecimals(0)
        self.solar_spin.setSuffix(" W/m²")
        amb_form.addRow("Solar Rad.:", self.solar_spin)

        self.daytime_check = QCheckBox("Daytime")
        self.daytime_check.setChecked(True)
        amb_form.addRow("", self.daytime_check)

        amb_group.setLayout(amb_form)
        layout.addWidget(amb_group)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        return {
            "wind_speed": self.wind_speed_spin.value(),
            "wind_direction": self.wind_dir_spin.value(),
            "reference_height": self.ref_height_spin.value(),
            "stability_class": self.stability_combo.currentText(),
            "terrain_type": self.terrain_combo.currentText(),
            "ambient_temperature": self.amb_temp_spin.value(),
            "ambient_pressure": self.amb_press_spin.value(),
            "relative_humidity": self.humidity_spin.value(),
            "cloud_cover": self.cloud_spin.value(),
            "solar_radiation": self.solar_spin.value(),
            "is_daytime": self.daytime_check.isChecked(),
        }

    def set_params(self, **kwargs):
        if "wind_speed" in kwargs:
            self.wind_speed_spin.setValue(kwargs["wind_speed"])
        if "wind_direction" in kwargs:
            self.wind_dir_spin.setValue(kwargs["wind_direction"])
        if "reference_height" in kwargs:
            self.ref_height_spin.setValue(kwargs["reference_height"])
        if "stability_class" in kwargs:
            idx = self.stability_combo.findText(kwargs["stability_class"])
            if idx >= 0:
                self.stability_combo.setCurrentIndex(idx)
        if "terrain_type" in kwargs:
            idx = self.terrain_combo.findText(kwargs["terrain_type"])
            if idx >= 0:
                self.terrain_combo.setCurrentIndex(idx)
        if "ambient_temperature" in kwargs:
            self.amb_temp_spin.setValue(kwargs["ambient_temperature"])
        if "ambient_pressure" in kwargs:
            self.amb_press_spin.setValue(kwargs["ambient_pressure"])
        if "relative_humidity" in kwargs:
            self.humidity_spin.setValue(kwargs["relative_humidity"])
        if "cloud_cover" in kwargs:
            self.cloud_spin.setValue(kwargs["cloud_cover"])
        if "solar_radiation" in kwargs:
            self.solar_spin.setValue(kwargs["solar_radiation"])
        if "is_daytime" in kwargs:
            self.daytime_check.setChecked(kwargs["is_daytime"])


# ══════════════════════════════════════════════════════════════════════════════
# Receptor Grid Tab
# ══════════════════════════════════════════════════════════════════════════════

class ReceptorGridTab(QWidget):
    """Receptor grid definition tab.

    Defines the 3D spatial grid for concentration evaluation:
      - x: Downwind distance (along wind direction)
      - y: Cross-wind distance (perpendicular to wind)
      - z: Vertical height above ground
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # X (downwind) range
        x_group = QGroupBox("Downwind Distance (X)")
        x_form = QFormLayout()

        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(10.0, 100000.0)
        self.x_min_spin.setValue(100.0)
        self.x_min_spin.setDecimals(0)
        self.x_min_spin.setSuffix(" m")
        x_form.addRow("X Min:", self.x_min_spin)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(100.0, 100000.0)
        self.x_max_spin.setValue(5000.0)
        self.x_max_spin.setDecimals(0)
        self.x_max_spin.setSuffix(" m")
        x_form.addRow("X Max:", self.x_max_spin)

        self.x_points_spin = QSpinBox()
        self.x_points_spin.setRange(10, 500)
        self.x_points_spin.setValue(51)
        x_form.addRow("X Points:", self.x_points_spin)

        x_group.setLayout(x_form)
        layout.addWidget(x_group)

        # Y (cross-wind) range
        y_group = QGroupBox("Cross-wind Distance (Y)")
        y_form = QFormLayout()

        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-50000.0, 0.0)
        self.y_min_spin.setValue(-500.0)
        self.y_min_spin.setDecimals(0)
        self.y_min_spin.setSuffix(" m")
        y_form.addRow("Y Min:", self.y_min_spin)

        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(0.0, 50000.0)
        self.y_max_spin.setValue(500.0)
        self.y_max_spin.setDecimals(0)
        self.y_max_spin.setSuffix(" m")
        y_form.addRow("Y Max:", self.y_max_spin)

        self.y_points_spin = QSpinBox()
        self.y_points_spin.setRange(10, 500)
        self.y_points_spin.setValue(51)
        y_form.addRow("Y Points:", self.y_points_spin)

        y_group.setLayout(y_form)
        layout.addWidget(y_group)

        # Z (vertical) range
        z_group = QGroupBox("Vertical Height (Z)")
        z_form = QFormLayout()

        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(0.0, 500.0)
        self.z_min_spin.setValue(0.0)
        self.z_min_spin.setDecimals(0)
        self.z_min_spin.setSuffix(" m")
        z_form.addRow("Z Min:", self.z_min_spin)

        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(1.0, 1000.0)
        self.z_max_spin.setValue(100.0)
        self.z_max_spin.setDecimals(0)
        self.z_max_spin.setSuffix(" m")
        z_form.addRow("Z Max:", self.z_max_spin)

        self.z_points_spin = QSpinBox()
        self.z_points_spin.setRange(2, 100)
        self.z_points_spin.setValue(21)
        z_form.addRow("Z Points:", self.z_points_spin)

        z_group.setLayout(z_form)
        layout.addWidget(z_group)

        # Quick presets
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()

        near_btn = QPushButton("Near-field (50-500m)")
        near_btn.clicked.connect(
            lambda: self._apply_preset(50, 500, 51, -100, 100, 41, 0, 50, 11)
        )
        preset_layout.addWidget(near_btn)

        mid_btn = QPushButton("Mid-field (100-5000m)")
        mid_btn.clicked.connect(
            lambda: self._apply_preset(100, 5000, 51, -500, 500, 51, 0, 200, 21)
        )
        preset_layout.addWidget(mid_btn)

        far_btn = QPushButton("Far-field (500-20000m)")
        far_btn.clicked.connect(
            lambda: self._apply_preset(500, 20000, 51, -2000, 2000, 51, 0, 500, 21)
        )
        preset_layout.addWidget(far_btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        layout.addStretch()

    def _apply_preset(
        self, x0, x1, nx, y0, y1, ny, z0, z1, nz
    ):
        self.x_min_spin.setValue(x0)
        self.x_max_spin.setValue(x1)
        self.x_points_spin.setValue(nx)
        self.y_min_spin.setValue(y0)
        self.y_max_spin.setValue(y1)
        self.y_points_spin.setValue(ny)
        self.z_min_spin.setValue(z0)
        self.z_max_spin.setValue(z1)
        self.z_points_spin.setValue(nz)

    def get_params(self) -> Dict[str, Any]:
        return {
            "grid_x_range": (
                self.x_min_spin.value(),
                self.x_max_spin.value(),
                self.x_points_spin.value(),
            ),
            "grid_y_range": (
                self.y_min_spin.value(),
                self.y_max_spin.value(),
                self.y_points_spin.value(),
            ),
            "grid_z_range": (
                self.z_min_spin.value(),
                self.z_max_spin.value(),
                self.z_points_spin.value(),
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Advanced Tab
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedTab(QWidget):
    """Advanced dispersion parameters.

    Includes chemical decay rate, dry deposition velocity, sampling time
    correction, and model selection override.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Decay
        decay_group = QGroupBox("Chemical Decay & Deposition")
        decay_form = QFormLayout()

        self.decay_rate_spin = QDoubleSpinBox()
        self.decay_rate_spin.setRange(0.0, 1.0)
        self.decay_rate_spin.setValue(0.0)
        self.decay_rate_spin.setDecimals(6)
        self.decay_rate_spin.setSuffix(" 1/s")
        self.decay_rate_spin.setToolTip(
            "First-order chemical decay rate. 0 = no decay.\n"
            "Typical values: 1e-4 to 1e-6 /s for many gases."
        )
        decay_form.addRow("Decay Rate (λ):", self.decay_rate_spin)

        self.deposition_spin = QDoubleSpinBox()
        self.deposition_spin.setRange(0.0, 1.0)
        self.deposition_spin.setValue(0.0)
        self.deposition_spin.setDecimals(4)
        self.deposition_spin.setSuffix(" m/s")
        self.deposition_spin.setToolTip(
            "Dry deposition velocity. 0 = no deposition.\n"
            "Typical: 0.001-0.05 m/s for reactive gases."
        )
        decay_form.addRow("Deposition (Vd):", self.deposition_spin)

        decay_group.setLayout(decay_form)
        layout.addWidget(decay_group)

        # Averaging
        avg_group = QGroupBox("Averaging Time")
        avg_form = QFormLayout()

        self.sampling_time_spin = QDoubleSpinBox()
        self.sampling_time_spin.setRange(1.0, 86400.0)
        self.sampling_time_spin.setValue(600.0)
        self.sampling_time_spin.setDecimals(0)
        self.sampling_time_spin.setSuffix(" s")
        self.sampling_time_spin.setToolTip(
            "Averaging time for concentration estimates.\n"
            "Standard Pasquill-Gifford coefficients are for 10 min (600 s).\n"
            "Longer averages account for plume meander."
        )
        avg_form.addRow("Sampling Time:", self.sampling_time_spin)

        self.ref_time_spin = QDoubleSpinBox()
        self.ref_time_spin.setRange(1.0, 86400.0)
        self.ref_time_spin.setValue(600.0)
        self.ref_time_spin.setDecimals(0)
        self.ref_time_spin.setSuffix(" s")
        avg_form.addRow("Reference Time:", self.ref_time_spin)

        avg_group.setLayout(avg_form)
        layout.addWidget(avg_group)

        # Puff-specific
        puff_group = QGroupBox("Puff Settings")
        puff_form = QFormLayout()

        self.puff_end_spin = QDoubleSpinBox()
        self.puff_end_spin.setRange(60.0, 86400.0)
        self.puff_end_spin.setValue(3600.0)
        self.puff_end_spin.setDecimals(0)
        self.puff_end_spin.setSuffix(" s")
        puff_form.addRow("Sim Duration:", self.puff_end_spin)

        self.puff_steps_spin = QSpinBox()
        self.puff_steps_spin.setRange(10, 500)
        self.puff_steps_spin.setValue(100)
        puff_form.addRow("Time Steps:", self.puff_steps_spin)

        self.num_puffs_spin = QSpinBox()
        self.num_puffs_spin.setRange(5, 100)
        self.num_puffs_spin.setValue(20)
        self.num_puffs_spin.setToolTip(
            "Number of puffs for finite-duration release superposition.\n"
            "More puffs = higher accuracy but slower calculation."
        )
        puff_form.addRow("Num. Puffs:", self.num_puffs_spin)

        puff_group.setLayout(puff_form)
        layout.addWidget(puff_group)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        return {
            "decay_rate": self.decay_rate_spin.value(),
            "deposition_velocity": self.deposition_spin.value(),
            "sampling_time": self.sampling_time_spin.value(),
            "reference_time": self.ref_time_spin.value(),
            "puff_time_end": self.puff_end_spin.value(),
            "puff_time_steps": self.puff_steps_spin.value(),
            "num_puffs": self.num_puffs_spin.value(),
        }
