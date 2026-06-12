"""
Rekarisk UI — Source Term Input Panel.

Provides a tabbed input form for all source term calculation types:
Orifice, Vessel Blowdown, Pipe Rupture, Relief Valve, and Pool Spill.

Uses PyQt6 for the UI framework.
"""

from __future__ import annotations

from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox,
    QPushButton, QFormLayout, QGroupBox, QCheckBox,
    QSplitter, QFrame, QTextEdit, QScrollArea, QSpinBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ══════════════════════════════════════════════════════════════════════════════
# Source Term Panel — Main Tabbed Input
# ══════════════════════════════════════════════════════════════════════════════

class SourceTermPanel(QWidget):
    """Main source term input panel with tabbed interface.

    Tabs:
      - Orifice: Liquid, gas, and two-phase orifice discharge
      - Vessel: Vessel blowdown / depressurization
      - Pipe: Pipe flow and rupture
      - PSV: Relief valve sizing (API 520)
      - Pool: Liquid pool spreading & evaporation
    """

    # Signals
    calculation_requested = pyqtSignal(str, dict)  # (calc_type, params_dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create tabs
        self._orifice_tab = OrificeTab()
        self._vessel_tab = VesselTab()
        self._pipe_tab = PipeTab()
        self._psv_tab = PSVTab()
        self._pool_tab = PoolTab()

        self.tabs.addTab(self._orifice_tab, "🔩 Orifice")
        self.tabs.addTab(self._vessel_tab, "🛢️ Vessel")
        self.tabs.addTab(self._pipe_tab, "🔧 Pipe")
        self.tabs.addTab(self._psv_tab, "🔒 PSV")
        self.tabs.addTab(self._pool_tab, "💧 Pool")

        layout.addWidget(self.tabs)

        # Connect signals
        self._orifice_tab.calculate_clicked.connect(
            lambda d: self.calculation_requested.emit("orifice", d))
        self._vessel_tab.calculate_clicked.connect(
            lambda d: self.calculation_requested.emit("vessel", d))
        self._pipe_tab.calculate_clicked.connect(
            lambda d: self.calculation_requested.emit("pipe", d))
        self._psv_tab.calculate_clicked.connect(
            lambda d: self.calculation_requested.emit("psv", d))
        self._pool_tab.calculate_clicked.connect(
            lambda d: self.calculation_requested.emit("pool", d))

    def get_current_tab_params(self) -> tuple[str, dict]:
        """Get parameters from the currently visible tab."""
        idx = self.tabs.currentIndex()
        tab_map = {
            0: ("orifice", self._orifice_tab),
            1: ("vessel", self._vessel_tab),
            2: ("pipe", self._pipe_tab),
            3: ("psv", self._psv_tab),
            4: ("pool", self._pool_tab),
        }
        calc_type, tab = tab_map.get(idx, ("orifice", self._orifice_tab))
        return calc_type, tab.get_params()

    def set_substance(self, substance) -> None:
        """Pre-fill the active tab from a Substance database entry."""
        idx = self.tabs.currentIndex()
        tab_map = {
            0: self._orifice_tab,
            1: self._vessel_tab,
            2: self._pipe_tab,
            3: self._psv_tab,
            4: self._pool_tab,
        }
        tab = tab_map.get(idx)
        if tab and hasattr(tab, 'set_substance'):
            tab.set_substance(substance)


# ══════════════════════════════════════════════════════════════════════════════
# Orifice Tab
# ══════════════════════════════════════════════════════════════════════════════

class OrificeTab(QWidget):
    """Orifice discharge input form."""

    calculate_clicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Fluid parameters group
        fluid_group = QGroupBox("Fluid Properties")
        fluid_form = QFormLayout()

        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["auto", "gas", "liquid", "two_phase"])
        fluid_form.addRow("Phase:", self.phase_combo)

        self.rho_input = self._make_spin(0.0, 10000.0, 1.2, 2, "kg/m³")
        fluid_form.addRow("Density ρ:", self.rho_input)

        self.mw_input = self._make_spin(0.001, 1.0, 0.029, 6, "kg/mol")
        fluid_form.addRow("Mol. Weight:", self.mw_input)

        self.k_input = self._make_spin(1.01, 2.0, 1.4, 3, "")
        fluid_form.addRow("Cp/Cv (k):", self.k_input)

        fluid_group.setLayout(fluid_form)
        scroll_layout.addWidget(fluid_group)

        # Orifice parameters group
        orifice_group = QGroupBox("Orifice Geometry & Conditions")
        orifice_form = QFormLayout()

        self.Cd_input = self._make_spin(0.1, 1.0, 0.62, 3, "")
        orifice_form.addRow("Cd:", self.Cd_input)

        self.d_hole_input = self._make_spin(0.001, 2.0, 0.025, 4, "m")
        orifice_form.addRow("Hole diameter:", self.d_hole_input)

        self.P_up_input = self._make_spin(1e3, 1e9, 5e5, 0, "Pa")
        orifice_form.addRow("P upstream:", self.P_up_input)

        self.P_down_input = self._make_spin(0.0, 1e9, 101325.0, 0, "Pa")
        orifice_form.addRow("P downstream:", self.P_down_input)

        self.T_input = self._make_spin(10.0, 3000.0, 300.0, 1, "K")
        orifice_form.addRow("Temperature:", self.T_input)

        self.head_input = self._make_spin(0.0, 100.0, 0.0, 2, "m")
        orifice_form.addRow("Liquid head:", self.head_input)

        self.duration_input = self._make_spin(0.0, 86400.0, 0.0, 0, "s")
        orifice_form.addRow("Duration:", self.duration_input)

        orifice_group.setLayout(orifice_form)
        scroll_layout.addWidget(orifice_group)

        # Calculate button
        btn_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Calculate Orifice Discharge")
        self.calc_btn.clicked.connect(self._on_calculate)
        btn_layout.addStretch()
        btn_layout.addWidget(self.calc_btn)
        scroll_layout.addLayout(btn_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _make_spin(self, vmin, vmax, default, decimals, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def _on_calculate(self):
        params = self.get_params()
        self.calculate_clicked.emit(params)

    def get_params(self) -> dict:
        return {
            "Cd": self.Cd_input.value(),
            "d_hole": self.d_hole_input.value(),
            "P_upstream": self.P_up_input.value(),
            "P_downstream": self.P_down_input.value(),
            "T": self.T_input.value(),
            "phase": self.phase_combo.currentText(),
            "rho": self.rho_input.value(),
            "molecular_weight": self.mw_input.value(),
            "cp_cv_ratio": self.k_input.value(),
            "h_liquid_head": self.head_input.value(),
            "duration": self.duration_input.value() if self.duration_input.value() > 0 else None,
        }

    def set_substance(self, substance) -> None:
        """Pre-fill fields from a Substance database entry."""
        is_gas = getattr(substance, 'is_gas_at_ambient', False)

        # Phase
        self.phase_combo.blockSignals(True)
        self.phase_combo.setCurrentText("gas" if is_gas else "liquid")
        self.phase_combo.blockSignals(False)

        # Density
        rho = substance.vapor_density if is_gas else substance.liquid_density
        if rho is not None:
            self.rho_input.setValue(rho)

        # Molecular weight (DB g/mol → widget kg/mol)
        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self.mw_input.setValue(mw / 1000.0)

        # Cp/Cv default
        self.k_input.setValue(1.4)


# ══════════════════════════════════════════════════════════════════════════════
# Vessel Blowdown Tab
# ══════════════════════════════════════════════════════════════════════════════

class VesselTab(QWidget):
    """Vessel blowdown input form."""

    calculate_clicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        sl = QVBoxLayout(w)

        # Vessel specs
        g1 = QGroupBox("Vessel Specifications")
        f1 = QFormLayout()
        self.V_input = self._spin(0.1, 1e6, 10.0, 1, "m³")
        f1.addRow("Volume:", self.V_input)
        self.A_wall_input = self._spin(0.1, 1e6, 25.0, 1, "m²")
        f1.addRow("Wall area:", self.A_wall_input)
        g1.setLayout(f1)
        sl.addWidget(g1)

        # Initial conditions
        g2 = QGroupBox("Initial Conditions")
        f2 = QFormLayout()
        self.P0_input = self._spin(1e3, 1e9, 6e5, 0, "Pa")
        f2.addRow("P initial:", self.P0_input)
        self.T0_input = self._spin(10.0, 3000.0, 300.0, 1, "K")
        f2.addRow("T initial:", self.T0_input)
        self.phase_vessel = QComboBox()
        self.phase_vessel.addItems(["gas", "two_phase"])
        f2.addRow("Phase:", self.phase_vessel)
        g2.setLayout(f2)
        sl.addWidget(g2)

        # Orifice
        g3 = QGroupBox("Orifice")
        f3 = QFormLayout()
        self.d_orifice = self._spin(0.001, 0.5, 0.025, 4, "m")
        f3.addRow("Orifice d:", self.d_orifice)
        self.cd_vessel = self._spin(0.1, 1.0, 0.62, 3, "")
        f3.addRow("Cd:", self.cd_vessel)
        g3.setLayout(f3)
        sl.addWidget(g3)

        # Simulation
        g4 = QGroupBox("Simulation Parameters")
        f4 = QFormLayout()
        self.t_max = self._spin(1.0, 86400.0, 60.0, 0, "s")
        f4.addRow("Max time:", self.t_max)
        self.p_target = self._spin(1000.0, 1e8, 101325.0, 0, "Pa")
        f4.addRow("P target:", self.p_target)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["api521", "rigorous"])
        f4.addRow("Mode:", self.mode_combo)
        g4.setLayout(f4)
        sl.addWidget(g4)

        # Fluid props
        g5 = QGroupBox("Fluid Properties")
        f5 = QFormLayout()
        self.mw_vessel = self._spin(0.001, 1.0, 0.029, 6, "kg/mol")
        f5.addRow("Mol. Weight:", self.mw_vessel)
        self.k_vessel = self._spin(1.01, 2.0, 1.4, 3, "")
        f5.addRow("Cp/Cv:", self.k_vessel)
        self.rho_l = self._spin(100.0, 5000.0, 1000.0, 1, "kg/m³")
        f5.addRow("Rho liquid:", self.rho_l)
        g5.setLayout(f5)
        sl.addWidget(g5)

        btn = QPushButton("Calculate Vessel Blowdown")
        btn.clicked.connect(lambda: self.calculate_clicked.emit(self.get_params()))
        bh = QHBoxLayout()
        bh.addStretch()
        bh.addWidget(btn)
        sl.addLayout(bh)
        sl.addStretch()

        scroll.setWidget(w)
        layout.addWidget(scroll)

    def _spin(self, vmin, vmax, default, decimals, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def get_params(self) -> dict:
        return {
            "V": self.V_input.value(),
            "A_wall": self.A_wall_input.value(),
            "P_initial": self.P0_input.value(),
            "T_initial": self.T0_input.value(),
            "orifice_d": self.d_orifice.value(),
            "Cd": self.cd_vessel.value(),
            "t_max": self.t_max.value(),
            "P_target": self.p_target.value(),
            "phase": self.phase_vessel.currentText(),
            "mode": self.mode_combo.currentText(),
            "molecular_weight": self.mw_vessel.value(),
            "cp_cv_ratio": self.k_vessel.value(),
            "rho_liquid": self.rho_l.value(),
        }

    def set_substance(self, substance) -> None:
        """Pre-fill fields from a Substance database entry."""
        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self.mw_vessel.setValue(mw / 1000.0)

        rho_l = getattr(substance, 'liquid_density', None)
        if rho_l is not None:
            self.rho_l.setValue(rho_l)


# ══════════════════════════════════════════════════════════════════════════════
# Pipe Tab
# ══════════════════════════════════════════════════════════════════════════════

class PipeTab(QWidget):
    """Pipe flow / rupture input form."""

    calculate_clicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        sl = QVBoxLayout(w)

        g1 = QGroupBox("Pipe Geometry")
        f1 = QFormLayout()
        self.L_input = self._spin(0.1, 1e6, 100.0, 1, "m")
        f1.addRow("Length:", self.L_input)
        self.D_input = self._spin(0.001, 5.0, 0.15, 4, "m")
        f1.addRow("Diameter:", self.D_input)
        self.eps_input = self._spin(1e-8, 0.01, 0.0000457, 6, "m")
        f1.addRow("Roughness:", self.eps_input)
        g1.setLayout(f1)
        sl.addWidget(g1)

        g2 = QGroupBox("Rupture / Flow")
        f2 = QFormLayout()
        self.rupture_combo = QComboBox()
        self.rupture_combo.addItems(["full_bore", "hole_in_pipe", "long_pipeline", "pipeline_blowdown"])
        self.rupture_combo.currentTextChanged.connect(self._on_rupture_type_changed)
        f2.addRow("Type:", self.rupture_combo)
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["gas", "liquid", "two_phase", "flashing_flow"])
        self.fluid_combo.currentTextChanged.connect(self._on_fluid_type_changed)
        f2.addRow("Fluid:", self.fluid_combo)
        self.P_up_pipe = self._spin(1e3, 1e9, 2e6, 0, "Pa")
        f2.addRow("P upstream:", self.P_up_pipe)
        self.P_down_pipe = self._spin(0.0, 1e9, 101325.0, 0, "Pa")
        f2.addRow("P downstream:", self.P_down_pipe)
        self.T_pipe = self._spin(10.0, 3000.0, 300.0, 1, "K")
        f2.addRow("Temperature:", self.T_pipe)
        self.d_hole_pipe = self._spin(0.001, 1.0, 0.025, 4, "m")
        f2.addRow("Hole d:", self.d_hole_pipe)
        self.cd_pipe = self._spin(0.1, 1.0, 0.62, 3, "")
        f2.addRow("Cd:", self.cd_pipe)
        g2.setLayout(f2)
        sl.addWidget(g2)

        g3 = QGroupBox("Fluid Properties")
        f3 = QFormLayout()
        self.rho_pipe = self._spin(0.1, 5000.0, 1.2, 2, "kg/m³")
        f3.addRow("Density:", self.rho_pipe)
        self.mu_pipe = self._spin(1e-7, 1.0, 1.8e-5, 7, "Pa·s")
        f3.addRow("Viscosity:", self.mu_pipe)
        self.mw_pipe = self._spin(0.001, 1.0, 0.029, 6, "kg/mol")
        f3.addRow("Mol. Weight:", self.mw_pipe)
        self.k_pipe = self._spin(1.01, 2.0, 1.4, 3, "")
        f3.addRow("Cp/Cv:", self.k_pipe)
        self.dynamic_eos_check = QCheckBox("Dynamic EOS (CoolProp)")
        self.dynamic_eos_check.setChecked(False)
        f3.addRow("", self.dynamic_eos_check)
        self.mole_frac_btn = QPushButton("Edit Composition...")
        self.mole_frac_btn.clicked.connect(self._edit_mole_fractions)
        self.mole_frac_btn.setEnabled(False)
        f3.addRow("", self.mole_frac_btn)
        g3.setLayout(f3)
        sl.addWidget(g3)

        # Advanced options for pipeline_blowdown
        self.g4_advanced = QGroupBox("Advanced (Pipeline Blowdown)")
        self.g4_advanced.setVisible(False)
        f4 = QFormLayout()
        self.n_segments_spin = QSpinBox()
        self.n_segments_spin.setRange(1, 100)
        self.n_segments_spin.setValue(10)
        f4.addRow("Segments:", self.n_segments_spin)
        self.wall_thickness_spin = self._spin(0.001, 0.1, 0.01, 3, "m")
        f4.addRow("Wall thickness:", self.wall_thickness_spin)
        self.wall_htc_spin = self._spin(0.0, 1000.0, 0.0, 1, "W/m²/K")
        f4.addRow("Wall HTC:", self.wall_htc_spin)
        self.g4_advanced.setLayout(f4)
        sl.addWidget(self.g4_advanced)

        btn = QPushButton("Calculate Pipe Flow")
        btn.clicked.connect(lambda: self.calculate_clicked.emit(self.get_params()))
        bh = QHBoxLayout()
        bh.addStretch()
        bh.addWidget(btn)
        sl.addLayout(bh)
        sl.addStretch()

        scroll.setWidget(w)
        layout.addWidget(scroll)

        # Initialize state
        self._mole_fractions = {}  # Dict[str, float]

    def _spin(self, vmin, vmax, default, decimals, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def _on_rupture_type_changed(self, value):
        """Handle rupture type change."""
        is_blowdown = value == "pipeline_blowdown"
        self.g4_advanced.setVisible(is_blowdown)

    def _on_fluid_type_changed(self, value):
        """Handle fluid type change."""
        is_dynamic = value in ("gas", "flashing_flow")
        self.mole_frac_btn.setEnabled(is_dynamic and self.dynamic_eos_check.isChecked())

    def _edit_mole_fractions(self):
        """Open dialog to edit mole fractions."""
        dialog = MoleFractionDialog(self._mole_fractions, self)
        if dialog.exec():
            self._mole_fractions = dialog.get_fractions()

    def get_params(self) -> dict:
        return {
            "L": self.L_input.value(),
            "D": self.D_input.value(),
            "roughness": self.eps_input.value(),
            "P_up": self.P_up_pipe.value(),
            "P_down": self.P_down_pipe.value(),
            "T": self.T_pipe.value(),
            "rupture_type": self.rupture_combo.currentText(),
            "fluid": self.fluid_combo.currentText(),
            "rho": self.rho_pipe.value(),
            "mu": self.mu_pipe.value(),
            "molecular_weight": self.mw_pipe.value(),
            "cp_cv_ratio": self.k_pipe.value(),
            "d_hole": self.d_hole_pipe.value(),
            "Cd": self.cd_pipe.value(),
            "dynamic_props": self.dynamic_eos_check.isChecked(),
            "mole_fractions": self._mole_fractions.copy() if self._mole_fractions else None,
            # Pipeline blowdown specific
            "n_segments": self.n_segments_spin.value(),
            "wall_thickness": self.wall_thickness_spin.value(),
            "wall_htc": self.wall_htc_spin.value(),
        }

    def set_substance(self, substance) -> None:
        """Pre-fill fields from a Substance database entry."""
        is_gas = getattr(substance, 'is_gas_at_ambient', False)

        rho = substance.vapor_density if is_gas else substance.liquid_density
        if rho is not None:
            self.rho_pipe.setValue(rho)

        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self.mw_pipe.setValue(mw / 1000.0)

        # Fluid type
        self.fluid_combo.blockSignals(True)
        self.fluid_combo.setCurrentText("gas" if is_gas else "liquid")
        self.fluid_combo.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
# PSV (Relief Valve) Tab
# ══════════════════════════════════════════════════════════════════════════════

class PSVTab(QWidget):
    """Relief valve sizing (API 520) input form."""

    calculate_clicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        sl = QVBoxLayout(w)

        g1 = QGroupBox("Relief Specifications")
        f1 = QFormLayout()
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "blocked_outlet", "fire_exposure", "thermal_expansion",
            "control_valve_failure", "cooling_water_failure",
            "external_pool_fire", "reflux_loss", "power_failure",
        ])
        f1.addRow("Scenario:", self.scenario_combo)
        self.fluid_psv = QComboBox()
        self.fluid_psv.addItems(["gas", "liquid", "steam", "two_phase"])
        f1.addRow("Fluid:", self.fluid_psv)
        self.pset_input = self._spin(1e3, 1e9, 1e6, 0, "Pa(g)")
        f1.addRow("Set pressure:", self.pset_input)
        self.pback_input = self._spin(0.0, 1e9, 0.0, 0, "Pa(g)")
        f1.addRow("Back pressure:", self.pback_input)
        self.trel_input = self._spin(50.0, 3000.0, 350.0, 1, "K")
        f1.addRow("T relieving:", self.trel_input)
        self.flow_input = self._spin(0.001, 1e6, 2.5, 3, "kg/s")
        f1.addRow("Flow rate:", self.flow_input)
        g1.setLayout(f1)
        sl.addWidget(g1)

        g2 = QGroupBox("Valve Configuration")
        f2 = QFormLayout()
        self.vtype_combo = QComboBox()
        self.vtype_combo.addItems(["conventional", "balanced_bellows", "pilot_operated"])
        f2.addRow("Valve type:", self.vtype_combo)
        self.op_input = self._spin(0.0, 100.0, 10.0, 1, "%")
        f2.addRow("Overpressure:", self.op_input)
        self.rd_check = QCheckBox("Rupture disk used")
        f2.addRow("", self.rd_check)
        g2.setLayout(f2)
        sl.addWidget(g2)

        g3 = QGroupBox("Fluid Properties")
        f3 = QFormLayout()
        self.mw_psv = self._spin(0.001, 1.0, 0.029, 6, "kg/mol")
        f3.addRow("Mol. Weight:", self.mw_psv)
        self.k_psv = self._spin(1.01, 2.0, 1.4, 3, "")
        f3.addRow("Cp/Cv:", self.k_psv)
        self.rho_psv = self._spin(100.0, 5000.0, 1000.0, 1, "kg/m³")
        f3.addRow("Rho liquid:", self.rho_psv)
        g3.setLayout(f3)
        sl.addWidget(g3)

        btn = QPushButton("Size Relief Valve")
        btn.clicked.connect(lambda: self.calculate_clicked.emit(self.get_params()))
        bh = QHBoxLayout()
        bh.addStretch()
        bh.addWidget(btn)
        sl.addLayout(bh)
        sl.addStretch()

        scroll.setWidget(w)
        layout.addWidget(scroll)

    def _spin(self, vmin, vmax, default, decimals, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def get_params(self) -> dict:
        return {
            "scenario": self.scenario_combo.currentText(),
            "P_set": self.pset_input.value(),
            "P_back": self.pback_input.value(),
            "T_relieving": self.trel_input.value(),
            "flow_rate": self.flow_input.value(),
            "fluid": self.fluid_psv.currentText(),
            "molecular_weight": self.mw_psv.value(),
            "cp_cv_ratio": self.k_psv.value(),
            "rho": self.rho_psv.value(),
            "valve_type": self.vtype_combo.currentText(),
            "overpressure_pct": self.op_input.value(),
            "rupture_disk_used": self.rd_check.isChecked(),
        }

    def set_substance(self, substance) -> None:
        """Pre-fill fields from a Substance database entry."""
        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self.mw_psv.setValue(mw / 1000.0)

        is_gas = getattr(substance, 'is_gas_at_ambient', False)
        if not is_gas:
            rho_l = getattr(substance, 'liquid_density', None)
            if rho_l is not None:
                self.rho_psv.setValue(rho_l)


# ══════════════════════════════════════════════════════════════════════════════
# Pool Tab
# ══════════════════════════════════════════════════════════════════════════════

class PoolTab(QWidget):
    """Pool spreading and evaporation input form."""

    calculate_clicked = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        sl = QVBoxLayout(w)

        g1 = QGroupBox("Spill Parameters")
        f1 = QFormLayout()
        self.substance_input = QLineEdit("generic")
        f1.addRow("Substance:", self.substance_input)
        self.mass_input = self._spin(1.0, 1e9, 1000.0, 1, "kg")
        f1.addRow("Spill mass:", self.mass_input)
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["land", "water", "concrete"])
        f1.addRow("Surface:", self.surface_combo)
        self.bunded_input = self._spin(0.0, 1e6, 0.0, 1, "m²")
        f1.addRow("Bunded area:", self.bunded_input)
        g1.setLayout(f1)
        sl.addWidget(g1)

        g2 = QGroupBox("Environmental")
        f2 = QFormLayout()
        self.tamb_input = self._spin(200.0, 400.0, 298.15, 1, "K")
        f2.addRow("T ambient:", self.tamb_input)
        self.ws_input = self._spin(0.5, 50.0, 3.0, 1, "m/s")
        f2.addRow("Wind speed:", self.ws_input)
        self.tmax_pool = self._spin(1.0, 3600.0, 120.0, 0, "s")
        f2.addRow("Max time:", self.tmax_pool)
        g2.setLayout(f2)
        sl.addWidget(g2)

        g3 = QGroupBox("Liquid Properties")
        f3 = QFormLayout()
        self.rho_pool = self._spin(100.0, 5000.0, 1000.0, 1, "kg/m³")
        f3.addRow("Density:", self.rho_pool)
        self.tboil_pool = self._spin(50.0, 1000.0, 373.15, 1, "K")
        f3.addRow("Boiling point:", self.tboil_pool)
        self.hfg_pool = self._spin(1000.0, 1e7, 2.26e6, 0, "J/kg")
        f3.addRow("Heat of vap.:", self.hfg_pool)
        self.pvap_pool = self._spin(0.0, 1e7, 3000.0, 0, "Pa")
        f3.addRow("Vapor pressure:", self.pvap_pool)
        self.mw_pool = self._spin(0.005, 1.0, 0.018, 6, "kg/mol")
        f3.addRow("Mol. Weight:", self.mw_pool)
        g3.setLayout(f3)
        sl.addWidget(g3)

        btn = QPushButton("Simulate Pool")
        btn.clicked.connect(lambda: self.calculate_clicked.emit(self.get_params()))
        bh = QHBoxLayout()
        bh.addStretch()
        bh.addWidget(btn)
        sl.addLayout(bh)
        sl.addStretch()

        scroll.setWidget(w)
        layout.addWidget(scroll)

    def _spin(self, vmin, vmax, default, decimals, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def get_params(self) -> dict:
        bunded = self.bunded_input.value()
        return {
            "substance": self.substance_input.text(),
            "spill_mass": self.mass_input.value(),
            "surface": self.surface_combo.currentText(),
            "bunded_area": bunded if bunded > 0 else None,
            "T_ambient": self.tamb_input.value(),
            "wind_speed": self.ws_input.value(),
            "t_max": self.tmax_pool.value(),
            "rho_l": self.rho_pool.value(),
            "boiling_point": self.tboil_pool.value(),
            "heat_of_vaporization": self.hfg_pool.value(),
            "vapor_pressure": self.pvap_pool.value(),
            "molecular_weight": self.mw_pool.value(),
        }

    def set_substance(self, substance) -> None:
        """Pre-fill fields from a Substance database entry."""
        self.substance_input.blockSignals(True)
        self.substance_input.setText(getattr(substance, 'name', 'generic'))
        self.substance_input.blockSignals(False)

        rho_l = getattr(substance, 'liquid_density', None)
        if rho_l is not None:
            self.rho_pool.setValue(rho_l)

        mw = getattr(substance, 'molecular_weight', None)
        if mw is not None:
            self.mw_pool.setValue(mw / 1000.0)

        bp = getattr(substance, 'normal_boiling_point', None)
        if bp is not None:
            self.tboil_pool.setValue(bp)


# ══════════════════════════════════════════════════════════════════════════════
# Mole Fraction Dialog (for CoolProp)
# ══════════════════════════════════════════════════════════════════════════════

class MoleFractionDialog(QDialog):
    """Dialog for editing mole fractions for CoolProp mixtures."""

    def __init__(self, fractions: Dict[str, float], parent=None):
        super().__init__(parent)
        self.fractions = fractions.copy() if fractions else {}
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Edit Mole Fractions")
        self.setModal(True)
        layout = QVBoxLayout(self)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Component", "Mole Fraction"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(10)
        layout.addWidget(self.table)

        # Add row button
        btn_add = QPushButton("Add Component")
        btn_add.clicked.connect(self._add_row)
        layout.addWidget(btn_add)

        # OK/Cancel buttons
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

        # Populate existing fractions
        row = 0
        for comp, frac in sorted(self.fractions.items()):
            if row >= self.table.rowCount():
                self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(comp))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(4)
            spin.setValue(frac)
            self.table.setCellWidget(row, 1, spin)
            row += 1

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(4)
        spin.setValue(0.1)
        self.table.setCellWidget(row, 1, spin)

    def get_fractions(self) -> Dict[str, float]:
        """Extract fractions from table."""
        fractions = {}
        for row in range(self.table.rowCount()):
            comp_item = self.table.item(row, 0)
            if comp_item is None or not comp_item.text().strip():
                continue
            comp = comp_item.text().strip()
            spin = self.table.cellWidget(row, 1)
            if spin is not None:
                frac = spin.value()
                if frac > 0:
                    fractions[comp] = frac
        return fractions
