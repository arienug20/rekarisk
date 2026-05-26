"""
Rekarisk UI — Fire Model Input Panel.

Provides a PyQt6 tabbed input panel for fire consequence calculations:
  - Pool Fire: Pool diameter, substance, burning rate, wind
  - Jet Fire: Orifice, discharge conditions, substance, wind, direction
  - BLEVE: Vessel mass, substance, optional SEP override
  - Flash Fire: Link to dispersion result, LFL/UFL values

Connected to the fire model calculators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QFormLayout, QGroupBox, QCheckBox,
    QProgressBar, QMessageBox, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from rekarisk.core.constants import P_ATM, T_0C, DEFAULT_AMBIENT_TEMP, DEFAULT_HUMIDITY


# ══════════════════════════════════════════════════════════════════════════════
# Substance List
# ══════════════════════════════════════════════════════════════════════════════

COMMON_SUBSTANCES = [
    "default", "methane", "ethane", "propane", "butane", "pentane",
    "hexane", "heptane", "octane", "gasoline", "kerosene", "diesel",
    "benzene", "toluene", "xylene", "methanol", "ethanol",
    "hydrogen", "ethylene", "propylene", "ammonia",
    "lpg", "lng", "crude_oil", "jp-4",
]

RELEASE_DIRECTIONS = ["horizontal", "vertical"]


# ══════════════════════════════════════════════════════════════════════════════
# Pool Fire Tab
# ══════════════════════════════════════════════════════════════════════════════

class PoolFireTab(QWidget):
    """Input form for pool fire calculations."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Fire Configuration ──
        fire_group = QGroupBox("🔥 Fire Configuration")
        fire_form = QFormLayout()

        self.pool_diameter = QDoubleSpinBox()
        self.pool_diameter.setRange(0.01, 500.0)
        self.pool_diameter.setValue(10.0)
        self.pool_diameter.setSuffix(" m")
        self.pool_diameter.setDecimals(2)
        self.pool_diameter.setToolTip("Diameter of the circular pool")
        fire_form.addRow("Pool Diameter:", self.pool_diameter)

        self.substance = QComboBox()
        self.substance.addItems(COMMON_SUBSTANCES)
        self.substance.setCurrentText("gasoline")
        self.substance.setToolTip("Select substance for auto property lookup")
        fire_form.addRow("Substance:", self.substance)

        fire_group.setLayout(fire_form)
        layout.addWidget(fire_group)

        # ── Advanced Parameters ──
        adv_group = QGroupBox("⚙️ Advanced Parameters")
        adv_form = QFormLayout()

        self.burning_rate = QDoubleSpinBox()
        self.burning_rate.setRange(0.0, 10.0)
        self.burning_rate.setValue(0.0)
        self.burning_rate.setSuffix(" kg/(m²·s)")
        self.burning_rate.setDecimals(5)
        self.burning_rate.setToolTip("Override auto burning rate. 0 = auto")
        adv_form.addRow("Burning Rate (0=auto):", self.burning_rate)

        self.heat_of_combustion = QDoubleSpinBox()
        self.heat_of_combustion.setRange(0.0, 200.0)
        self.heat_of_combustion.setValue(0.0)
        self.heat_of_combustion.setSuffix(" MJ/kg")
        self.heat_of_combustion.setDecimals(2)
        self.heat_of_combustion.setToolTip("Override heat of combustion. 0 = auto")
        adv_form.addRow("ΔHc (0=auto):", self.heat_of_combustion)

        self.radiative_fraction = QDoubleSpinBox()
        self.radiative_fraction.setRange(0.05, 0.60)
        self.radiative_fraction.setValue(0.35)
        self.radiative_fraction.setDecimals(3)
        self.radiative_fraction.setSingleStep(0.05)
        self.radiative_fraction.setToolTip("Fraction of heat radiated (0.15-0.40 typical)")
        adv_form.addRow("Radiative Fraction χᵣ:", self.radiative_fraction)

        adv_group.setLayout(adv_form)
        layout.addWidget(adv_group)

        # ── Environment ──
        env_group = QGroupBox("🌤️ Environmental Conditions")
        env_form = QFormLayout()

        self.wind_speed = QDoubleSpinBox()
        self.wind_speed.setRange(0.0, 25.0)
        self.wind_speed.setValue(3.0)
        self.wind_speed.setSuffix(" m/s")
        self.wind_speed.setDecimals(1)
        self.wind_speed.setToolTip("Wind speed at 10 m height")
        env_form.addRow("Wind Speed:", self.wind_speed)

        self.ambient_temp = QDoubleSpinBox()
        self.ambient_temp.setRange(233.15, 323.15)
        self.ambient_temp.setValue(DEFAULT_AMBIENT_TEMP)
        self.ambient_temp.setSuffix(" K")
        self.ambient_temp.setDecimals(1)
        self.ambient_temp.setToolTip("Ambient air temperature")
        env_form.addRow("Ambient Temperature:", self.ambient_temp)

        self.humidity = QDoubleSpinBox()
        self.humidity.setRange(0.0, 100.0)
        self.humidity.setValue(DEFAULT_HUMIDITY)
        self.humidity.setSuffix(" %")
        self.humidity.setDecimals(0)
        self.humidity.setToolTip("Relative humidity")
        env_form.addRow("Relative Humidity:", self.humidity)

        env_group.setLayout(env_form)
        layout.addWidget(env_group)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        """Collect pool fire parameters."""
        br = self.burning_rate.value()
        dhc = self.heat_of_combustion.value()
        return {
            "pool_diameter": self.pool_diameter.value(),
            "substance": self.substance.currentText(),
            "burning_rate": br if br > 0 else None,
            "heat_of_combustion": dhc * 1e6 if dhc > 0 else None,
            "radiative_fraction": self.radiative_fraction.value(),
            "wind_speed": self.wind_speed.value(),
            "ambient_temperature": self.ambient_temp.value(),
            "relative_humidity": self.humidity.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Jet Fire Tab
# ══════════════════════════════════════════════════════════════════════════════

class JetFireTab(QWidget):
    """Input form for jet fire calculations."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Release Configuration ──
        rel_group = QGroupBox("💨 Release Configuration")
        rel_form = QFormLayout()

        self.orifice_diameter = QDoubleSpinBox()
        self.orifice_diameter.setRange(0.001, 2.0)
        self.orifice_diameter.setValue(0.05)
        self.orifice_diameter.setSuffix(" m")
        self.orifice_diameter.setDecimals(4)
        self.orifice_diameter.setToolTip("Orifice/hole diameter")
        rel_form.addRow("Orifice Diameter:", self.orifice_diameter)

        self.discharge_velocity = QDoubleSpinBox()
        self.discharge_velocity.setRange(0.1, 1000.0)
        self.discharge_velocity.setValue(100.0)
        self.discharge_velocity.setSuffix(" m/s")
        self.discharge_velocity.setDecimals(1)
        self.discharge_velocity.setToolTip("Discharge velocity at orifice exit")
        rel_form.addRow("Discharge Velocity:", self.discharge_velocity)

        self.mass_flow_rate = QDoubleSpinBox()
        self.mass_flow_rate.setRange(0.0, 1000.0)
        self.mass_flow_rate.setValue(0.0)
        self.mass_flow_rate.setSuffix(" kg/s")
        self.mass_flow_rate.setDecimals(4)
        self.mass_flow_rate.setToolTip("Mass flow rate. 0 = estimate from velocity and area")
        rel_form.addRow("Mass Flow Rate (0=auto):", self.mass_flow_rate)

        self.jet_release_dir = QComboBox()
        self.jet_release_dir.addItems(RELEASE_DIRECTIONS)
        self.jet_release_dir.setCurrentText("horizontal")
        self.jet_release_dir.setToolTip("Direction of the jet release")
        rel_form.addRow("Release Direction:", self.jet_release_dir)

        rel_group.setLayout(rel_form)
        layout.addWidget(rel_group)

        # ── Substance ──
        sub_group = QGroupBox("🧪 Substance Properties")
        sub_form = QFormLayout()

        self.jet_substance = QComboBox()
        self.jet_substance.addItems(COMMON_SUBSTANCES)
        self.jet_substance.setCurrentText("propane")
        sub_form.addRow("Substance:", self.jet_substance)

        self.jet_dhc = QDoubleSpinBox()
        self.jet_dhc.setRange(0.0, 200.0)
        self.jet_dhc.setValue(0.0)
        self.jet_dhc.setSuffix(" MJ/kg")
        self.jet_dhc.setDecimals(2)
        self.jet_dhc.setToolTip("Heat of combustion. 0 = auto")
        sub_form.addRow("ΔHc (0=auto):", self.jet_dhc)

        self.jet_chi_r = QDoubleSpinBox()
        self.jet_chi_r.setRange(0.05, 0.60)
        self.jet_chi_r.setValue(0.30)
        self.jet_chi_r.setDecimals(3)
        self.jet_chi_r.setSingleStep(0.05)
        sub_form.addRow("Radiative Fraction χᵣ:", self.jet_chi_r)

        self.discharge_density = QDoubleSpinBox()
        self.discharge_density.setRange(0.0, 100.0)
        self.discharge_density.setValue(0.0)
        self.discharge_density.setSuffix(" kg/m³")
        self.discharge_density.setDecimals(3)
        self.discharge_density.setToolTip("Density at orifice. 0 = estimate from ideal gas")
        sub_form.addRow("Discharge Density (0=auto):", self.discharge_density)

        sub_group.setLayout(sub_form)
        layout.addWidget(sub_group)

        # ── Environment ──
        env_group = QGroupBox("🌤️ Environmental Conditions")
        env_form = QFormLayout()

        self.jet_wind = QDoubleSpinBox()
        self.jet_wind.setRange(0.0, 25.0)
        self.jet_wind.setValue(3.0)
        self.jet_wind.setSuffix(" m/s")
        env_form.addRow("Wind Speed:", self.jet_wind)

        self.jet_temp = QDoubleSpinBox()
        self.jet_temp.setRange(233.15, 323.15)
        self.jet_temp.setValue(DEFAULT_AMBIENT_TEMP)
        self.jet_temp.setSuffix(" K")
        env_form.addRow("Ambient Temperature:", self.jet_temp)

        self.jet_humidity = QDoubleSpinBox()
        self.jet_humidity.setRange(0.0, 100.0)
        self.jet_humidity.setValue(DEFAULT_HUMIDITY)
        self.jet_humidity.setSuffix(" %")
        env_form.addRow("Relative Humidity:", self.jet_humidity)

        env_group.setLayout(env_form)
        layout.addWidget(env_group)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        """Collect jet fire parameters."""
        mdot = self.mass_flow_rate.value()
        dhc = self.jet_dhc.value()
        rho = self.discharge_density.value()
        return {
            "orifice_diameter": self.orifice_diameter.value(),
            "discharge_velocity": self.discharge_velocity.value(),
            "mass_flow_rate": mdot if mdot > 0 else None,
            "substance": self.jet_substance.currentText(),
            "heat_of_combustion": dhc * 1e6 if dhc > 0 else None,
            "radiative_fraction": self.jet_chi_r.value(),
            "wind_speed": self.jet_wind.value(),
            "release_direction": self.jet_release_dir.currentText(),
            "ambient_temperature": self.jet_temp.value(),
            "relative_humidity": self.jet_humidity.value(),
            "discharge_density": rho if rho > 0 else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BLEVE Tab
# ══════════════════════════════════════════════════════════════════════════════

class BLEVETab(QWidget):
    """Input form for BLEVE / fireball calculations."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Vessel Configuration ──
        vessel_group = QGroupBox("🛢️ Vessel Configuration")
        vessel_form = QFormLayout()

        self.vessel_mass = QDoubleSpinBox()
        self.vessel_mass.setRange(1.0, 1_000_000.0)
        self.vessel_mass.setValue(1000.0)
        self.vessel_mass.setSuffix(" kg")
        self.vessel_mass.setDecimals(1)
        self.vessel_mass.setToolTip("Mass of contents released in BLEVE")
        vessel_form.addRow("Vessel Mass:", self.vessel_mass)

        self.bleve_substance = QComboBox()
        self.bleve_substance.addItems(COMMON_SUBSTANCES)
        self.bleve_substance.setCurrentText("propane")
        self.bleve_substance.setToolTip("Select substance")
        vessel_form.addRow("Substance:", self.bleve_substance)

        vessel_group.setLayout(vessel_form)
        layout.addWidget(vessel_group)

        # ── Advanced Parameters ──
        adv_group = QGroupBox("⚙️ Advanced Parameters")
        adv_form = QFormLayout()

        self.bleve_dhc = QDoubleSpinBox()
        self.bleve_dhc.setRange(0.0, 200.0)
        self.bleve_dhc.setValue(0.0)
        self.bleve_dhc.setSuffix(" MJ/kg")
        self.bleve_dhc.setDecimals(2)
        self.bleve_dhc.setToolTip("Heat of combustion. 0 = auto")
        adv_form.addRow("ΔHc (0=auto):", self.bleve_dhc)

        self.bleve_chi_r = QDoubleSpinBox()
        self.bleve_chi_r.setRange(0.10, 0.50)
        self.bleve_chi_r.setValue(0.30)
        self.bleve_chi_r.setDecimals(3)
        self.bleve_chi_r.setSingleStep(0.05)
        adv_form.addRow("Radiative Fraction χᵣ:", self.bleve_chi_r)

        self.sep_override = QDoubleSpinBox()
        self.sep_override.setRange(0.0, 500.0)
        self.sep_override.setValue(0.0)
        self.sep_override.setSuffix(" kW/m²")
        self.sep_override.setDecimals(1)
        self.sep_override.setToolTip("Override SEP. 0 = auto-calculate (200-350 typical)")
        adv_form.addRow("SEP Override (0=auto):", self.sep_override)

        adv_group.setLayout(adv_form)
        layout.addWidget(adv_group)

        # ── Environment ──
        env_group = QGroupBox("🌤️ Environmental Conditions")
        env_form = QFormLayout()

        self.bleve_temp = QDoubleSpinBox()
        self.bleve_temp.setRange(233.15, 323.15)
        self.bleve_temp.setValue(DEFAULT_AMBIENT_TEMP)
        self.bleve_temp.setSuffix(" K")
        env_form.addRow("Ambient Temperature:", self.bleve_temp)

        self.bleve_humidity = QDoubleSpinBox()
        self.bleve_humidity.setRange(0.0, 100.0)
        self.bleve_humidity.setValue(DEFAULT_HUMIDITY)
        self.bleve_humidity.setSuffix(" %")
        env_form.addRow("Relative Humidity:", self.bleve_humidity)

        env_group.setLayout(env_form)
        layout.addWidget(env_group)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        """Collect BLEVE parameters."""
        dhc = self.bleve_dhc.value()
        sep = self.sep_override.value()
        return {
            "vessel_mass": self.vessel_mass.value(),
            "substance": self.bleve_substance.currentText(),
            "heat_of_combustion": dhc * 1e6 if dhc > 0 else None,
            "radiative_fraction": self.bleve_chi_r.value(),
            "sep_override": sep if sep > 0 else None,
            "ambient_temperature": self.bleve_temp.value(),
            "relative_humidity": self.bleve_humidity.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Flash Fire Tab
# ══════════════════════════════════════════════════════════════════════════════

class FlashFireTab(QWidget):
    """Input form for flash fire calculations."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Dispersion Link ──
        disp_group = QGroupBox("🌫️ Dispersion Data Source")
        disp_form = QFormLayout()

        self.use_synthetic = QCheckBox("Use synthetic concentration field")
        self.use_synthetic.setChecked(True)
        self.use_synthetic.setToolTip(
            "Generate a synthetic Gaussian concentration field for testing.\n"
            "Uncheck to link from an existing dispersion result."
        )
        disp_form.addRow(self.use_synthetic)

        self.synthetic_release_rate = QDoubleSpinBox()
        self.synthetic_release_rate.setRange(0.001, 100.0)
        self.synthetic_release_rate.setValue(1.0)
        self.synthetic_release_rate.setSuffix(" kg/s")
        self.synthetic_release_rate.setDecimals(3)
        disp_form.addRow("Release Rate (synthetic):", self.synthetic_release_rate)

        disp_group.setLayout(disp_form)
        layout.addWidget(disp_group)

        # ── Flammability Limits ──
        flam_group = QGroupBox("🔥 Flammability Limits")
        flam_form = QFormLayout()

        self.ff_substance = QComboBox()
        self.ff_substance.addItems(COMMON_SUBSTANCES)
        self.ff_substance.setCurrentText("propane")
        self.ff_substance.setToolTip("Select substance for auto LFL/UFL")
        flam_form.addRow("Substance:", self.ff_substance)

        self.lfl_value = QDoubleSpinBox()
        self.lfl_value.setRange(0.01, 25.0)
        self.lfl_value.setValue(0.0)
        self.lfl_value.setSuffix(" % vol")
        self.lfl_value.setDecimals(2)
        self.lfl_value.setToolTip("Lower Flammable Limit. 0 = auto from substance")
        flam_form.addRow("LFL (0=auto):", self.lfl_value)

        self.ufl_value = QDoubleSpinBox()
        self.ufl_value.setRange(0.01, 80.0)
        self.ufl_value.setValue(0.0)
        self.ufl_value.setSuffix(" % vol")
        self.ufl_value.setDecimals(2)
        self.ufl_value.setToolTip("Upper Flammable Limit. 0 = auto from substance")
        flam_form.addRow("UFL (0=auto):", self.ufl_value)

        self.sep_flash = QDoubleSpinBox()
        self.sep_flash.setRange(50.0, 500.0)
        self.sep_flash.setValue(173.0)
        self.sep_flash.setSuffix(" kW/m²")
        self.sep_flash.setDecimals(1)
        self.sep_flash.setToolTip("Surface emissive power for flash fire (150-200 typical)")
        flam_form.addRow("SEP Flash:", self.sep_flash)

        flam_group.setLayout(flam_form)
        layout.addWidget(flam_group)

        # Info label
        info_label = QLabel(
            "Flash fire analysis determines the flammable cloud extent from "
            "dispersion results. The LFL contour defines the outer boundary "
            "of the flammable cloud. Any ignition source within this boundary "
            "may cause a flash fire."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 8px;")
        layout.addWidget(info_label)

        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        """Collect flash fire parameters."""
        lfl = self.lfl_value.value()
        ufl = self.ufl_value.value()
        return {
            "substance": self.ff_substance.currentText(),
            "lfl": lfl if lfl > 0 else None,
            "ufl": ufl if ufl > 0 else None,
            "sep_flash": self.sep_flash.value(),
            "use_synthetic": self.use_synthetic.isChecked(),
            "synthetic_release_rate": self.synthetic_release_rate.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main Fire Panel
# ══════════════════════════════════════════════════════════════════════════════

class FirePanel(QWidget):
    """Main fire calculation input panel with tabbed interface.

    Tabs:
      - Pool Fire: Circular pool fire thermal radiation
      - Jet Fire: Jet fire thermal radiation (API 521)
      - BLEVE: BLEVE / fireball thermal radiation
      - Flash Fire: Flash fire envelope from dispersion

    Signals:
        calculation_requested: Emitted with (model_type, params_dict)
            when the Run button is clicked.
    """

    calculation_requested = pyqtSignal(str, dict)

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
        self._pool_tab = PoolFireTab()
        self._jet_tab = JetFireTab()
        self._bleve_tab = BLEVETab()
        self._flash_tab = FlashFireTab()

        self.tabs.addTab(self._pool_tab, "🔥 Pool Fire")
        self.tabs.addTab(self._jet_tab, "💨 Jet Fire")
        self.tabs.addTab(self._bleve_tab, "💥 BLEVE")
        self.tabs.addTab(self._flash_tab, "🌫️ Flash Fire")

        layout.addWidget(self.tabs)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Run button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.run_button = QPushButton("🚀 Run Fire Calculation")
        self.run_button.setMinimumHeight(36)
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D32F2F; }"
        )
        self.run_button.clicked.connect(self._on_run)
        btn_layout.addWidget(self.run_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_run(self):
        """Collect parameters from current tab and emit signal."""
        current_idx = self.tabs.currentIndex()
        tab_names = ["pool_fire", "jet_fire", "bleve", "flash_fire"]
        model_type = tab_names[current_idx]

        if model_type == "pool_fire":
            params = self._pool_tab.get_params()
        elif model_type == "jet_fire":
            params = self._jet_tab.get_params()
        elif model_type == "bleve":
            params = self._bleve_tab.get_params()
        elif model_type == "flash_fire":
            params = self._flash_tab.get_params()
        else:
            return

        # Validate
        errors = self._validate_params(model_type, params)
        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(errors))
            return

        self.calculation_requested.emit(model_type, params)

    def _validate_params(self, model_type: str, params: Dict[str, Any]) -> list:
        """Validate input parameters before calculation."""
        errors = []

        if model_type == "pool_fire":
            if params["pool_diameter"] <= 0:
                errors.append("Pool diameter must be positive")
            if params["radiative_fraction"] <= 0 or params["radiative_fraction"] > 0.6:
                errors.append("Radiative fraction must be between 0.05 and 0.60")
        elif model_type == "jet_fire":
            if params["orifice_diameter"] <= 0:
                errors.append("Orifice diameter must be positive")
            if params["discharge_velocity"] <= 0:
                errors.append("Discharge velocity must be positive")
            mdot = params.get("mass_flow_rate")
            if (mdot is None or mdot <= 0) and params.get("discharge_density") is None:
                errors.append("Provide mass flow rate or discharge density")
        elif model_type == "bleve":
            if params["vessel_mass"] <= 0:
                errors.append("Vessel mass must be positive")
        elif model_type == "flash_fire":
            pass  # Flash fire can use synthetic data

        return errors

    def set_progress_visible(self, visible: bool):
        """Show or hide the progress bar."""
        self.progress.setVisible(visible)

    def set_progress_value(self, value: int):
        """Set progress bar value (0-100)."""
        self.progress.setValue(value)
