"""
Rekarisk UI — Explosion Input Panel.

Provides a PyQt6 tabbed input panel for explosion consequence analysis:
  - Tab 1: TNT Equivalency — mass, efficiency, substance lookup
  - Tab 2: TNO Multi-Energy — confinement, congestion, blast strength
  - Tab 3: BST — reactivity, confinement, congestion
  - Shared: distance range input and run button
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QFormLayout, QGroupBox, QCheckBox,
    QSlider, QProgressBar, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ──────────────────────────────────────────────────────────────────────────────
# Model options
# ──────────────────────────────────────────────────────────────────────────────

CONFINEMENT_CLASSES = ["1D", "2D", "3D"]
CONGESTION_LEVELS = ["low", "medium", "high"]
TNO_CONFINEMENT_CLASSES = ["none", "1D", "2D", "3D"]
REACTIVITY_LEVELS = ["high", "medium", "low"]

CONFINEMENT_DESC = {
    "none": "No confinement — open field",
    "1D": "1D — pipe rack, narrow alley",
    "2D": "2D — platform, multi-story deck",
    "3D": "3D — enclosed space, vessel",
}

CONGESTION_DESC = {
    "low": "Sparse equipment, few obstacles",
    "medium": "Typical process plant",
    "high": "Densely packed, high blockage",
}

REACTIVITY_DESC = {
    "high": "H₂, C₂H₂, ethylene oxide",
    "medium": "Most hydrocarbons (C₁–C₈)",
    "low": "CH₄, NH₃, CO",
}

# Substance database for quick ΔHc lookup
SUBSTANCE_LIST = [
    ("methane", 55.5e6, "low"),
    ("ethane", 51.9e6, "medium"),
    ("propane", 50.35e6, "medium"),
    ("n-butane", 49.5e6, "medium"),
    ("n-pentane", 49.0e6, "medium"),
    ("n-hexane", 48.4e6, "medium"),
    ("n-heptane", 48.1e6, "medium"),
    ("ethylene", 50.3e6, "medium"),
    ("propylene", 48.9e6, "medium"),
    ("benzene", 42.0e6, "medium"),
    ("toluene", 42.5e6, "medium"),
    ("methanol", 22.7e6, "medium"),
    ("ethanol", 29.7e6, "medium"),
    ("acetone", 30.8e6, "medium"),
    ("hydrogen", 141.8e6, "high"),
    ("acetylene", 49.9e6, "high"),
    ("ammonia", 22.5e6, "low"),
    ("carbon_monoxide", 10.1e6, "low"),
    ("gasoline", 46.4e6, "medium"),
    ("kerosene", 46.2e6, "medium"),
    ("diesel", 45.5e6, "medium"),
    ("lpg", 49.6e6, "medium"),
    ("lng", 55.0e6, "medium"),
    ("natural_gas", 50.0e6, "medium"),
    ("hydrogen_sulfide", 16.5e6, "low"),
    ("crude_oil", 44.0e6, "medium"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Main Panel
# ══════════════════════════════════════════════════════════════════════════════

class ExplosionPanel(QWidget):
    """Main explosion input panel with 3 method tabs.

    Tabs:
      - TNT Equivalency: Mass, efficiency, substance lookup
      - TNO Multi-Energy: Confinement, congestion, blast strength
      - BST: Reactivity, confinement, congestion

    Signals:
        calculation_requested: Emitted with full params dict when Run clicked.
    """

    calculation_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Method tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._tnt_tab = TNTInputTab()
        self._tno_tab = TNOInputTab()
        self._bst_tab = BSTInputTab()

        self.tabs.addTab(self._tnt_tab, "🧨 TNT Equivalency")
        self.tabs.addTab(self._tno_tab, "📊 TNO Multi-Energy")
        self.tabs.addTab(self._bst_tab, "🔥 Baker-Strehlow-Tang")

        layout.addWidget(self.tabs)

        # Distance range (shared)
        dist_group = QGroupBox("📍 Distance Range")
        dist_layout = QHBoxLayout()

        dist_layout.addWidget(QLabel("Min:"))
        self.dist_min_spin = QDoubleSpinBox()
        self.dist_min_spin.setRange(1.0, 1000.0)
        self.dist_min_spin.setValue(10.0)
        self.dist_min_spin.setDecimals(1)
        self.dist_min_spin.setSuffix(" m")
        dist_layout.addWidget(self.dist_min_spin)

        dist_layout.addWidget(QLabel("Max:"))
        self.dist_max_spin = QDoubleSpinBox()
        self.dist_max_spin.setRange(10.0, 50000.0)
        self.dist_max_spin.setValue(1000.0)
        self.dist_max_spin.setDecimals(0)
        self.dist_max_spin.setSuffix(" m")
        dist_layout.addWidget(self.dist_max_spin)

        dist_layout.addWidget(QLabel("Points:"))
        self.dist_points_spin = QSpinBox()
        self.dist_points_spin.setRange(20, 500)
        self.dist_points_spin.setValue(100)
        dist_layout.addWidget(self.dist_points_spin)

        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Run button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.run_all_check = QCheckBox("Run all models simultaneously")
        self.run_all_check.setChecked(True)
        btn_layout.addWidget(self.run_all_check)

        btn_layout.addSpacing(16)

        self.run_button = QPushButton("💥 Run Explosion Analysis")
        self.run_button.setMinimumHeight(36)
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #E64A19; }"
        )
        self.run_button.clicked.connect(self._on_run)
        btn_layout.addWidget(self.run_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_run(self):
        """Collect parameters and emit calculation request."""
        params = self.get_all_params()
        errors = []

        mass = params.get("mass_flammable", 0)
        delta_hc = params.get("heat_of_combustion", 0)
        if mass <= 0:
            errors.append("Mass of flammable material must be > 0 kg")
        if delta_hc is None or delta_hc <= 0:
            errors.append("Heat of combustion must be provided or substance selected")

        d_min = params.get("dist_min", 10)
        d_max = params.get("dist_max", 1000)
        if d_max <= d_min:
            errors.append("Distance max must be greater than min")

        run_all = params.get("run_all", True)
        active_tab = self.tabs.currentIndex()
        tnt_enabled = run_all or active_tab == 0
        tno_enabled = run_all or active_tab == 1
        bst_enabled = run_all or active_tab == 2

        if errors:
            QMessageBox.warning(
                self, "Validation Error",
                "Please fix the following:\n• " + "\n• ".join(errors)
            )
            return

        params["tnt_enabled"] = tnt_enabled
        params["tno_enabled"] = tno_enabled
        params["bst_enabled"] = bst_enabled

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_button.setEnabled(False)

        self.calculation_requested.emit(params)

    def on_calculation_complete(self):
        """Reset UI after calculation."""
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)

    def get_all_params(self) -> Dict[str, Any]:
        """Collect all parameters from all tabs + shared."""
        params = {
            "run_all": self.run_all_check.isChecked(),
            "dist_min": self.dist_min_spin.value(),
            "dist_max": self.dist_max_spin.value(),
            "dist_points": self.dist_points_spin.value(),
        }
        params.update(self._tnt_tab.get_params())
        params.update(self._tno_tab.get_params())
        params.update(self._bst_tab.get_params())
        return params


# ══════════════════════════════════════════════════════════════════════════════
# Shared Substance Selector
# ══════════════════════════════════════════════════════════════════════════════

class SubstanceSelectorGroup(QGroupBox):
    """Group box for substance selection with auto-lookup of ΔHc."""

    substance_changed = pyqtSignal(str, float, str)  # name, ΔHc, reactivity

    def __init__(self, parent: QWidget | None = None):
        super().__init__("🧪 Substance", parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        self.substance_combo = QComboBox()
        self.substance_combo.setEditable(True)
        self.substance_combo.addItem("— Custom —")
        for name, delta_hc, reactivity in SUBSTANCE_LIST:
            self.substance_combo.addItem(
                f"{name}  (ΔHc={delta_hc/1e6:.1f} MJ/kg)",
                (name, delta_hc, reactivity)
            )
        self.substance_combo.currentIndexChanged.connect(self._on_substance_changed)
        layout.addRow("Substance:", self.substance_combo)

        self.delta_hc_spin = QDoubleSpinBox()
        self.delta_hc_spin.setRange(0.1, 200.0)
        self.delta_hc_spin.setValue(50.35)
        self.delta_hc_spin.setDecimals(2)
        self.delta_hc_spin.setSuffix(" MJ/kg")
        layout.addRow("ΔHc:", self.delta_hc_spin)

        self.reactivity_label = QLabel("medium")
        self.reactivity_label.setStyleSheet("color: #555; font-style: italic;")
        layout.addRow("Reactivity:", self.reactivity_label)

    def _on_substance_changed(self, idx: int):
        if idx <= 0:
            return

        data = self.substance_combo.currentData()
        if data:
            name, delta_hc, reactivity = data
            self.delta_hc_spin.setValue(delta_hc / 1e6)
            self.reactivity_label.setText(reactivity)
            self.substance_changed.emit(name, delta_hc, reactivity)

    def get_substance_name(self) -> Optional[str]:
        idx = self.substance_combo.currentIndex()
        if idx <= 0:
            return None
        text = self.substance_combo.currentText()
        return text.split("  (")[0] if "  (" in text else text.strip()

    def get_delta_hc(self) -> float:
        return self.delta_hc_spin.value() * 1e6

    def get_reactivity(self) -> str:
        return self.reactivity_label.text()

    def set_substance(self, name: str, delta_hc: float):
        """Set substance from external data."""
        for i in range(1, self.substance_combo.count()):
            data = self.substance_combo.itemData(i)
            if data and data[0] == name:
                self.substance_combo.setCurrentIndex(i)
                return
        self.substance_combo.setCurrentText(name)
        self.delta_hc_spin.setValue(delta_hc / 1e6)


# ══════════════════════════════════════════════════════════════════════════════
# TNT Equivalency Tab
# ══════════════════════════════════════════════════════════════════════════════

class TNTInputTab(QWidget):
    """TNT equivalency input parameters."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Substance selector
        self.substance_group = SubstanceSelectorGroup()
        layout.addWidget(self.substance_group)

        # Mass input
        mass_group = QGroupBox("📦 Release Parameters")
        mass_form = QFormLayout()

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.1, 1e9)
        self.mass_spin.setValue(1000.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSuffix(" kg")
        mass_form.addRow("Mass Flammable:", self.mass_spin)

        # Connected to substance ΔHc
        self.substance_group.substance_changed.connect(
            lambda n, dhc, r: self._update_info()
        )
        self.mass_spin.valueChanged.connect(lambda v: self._update_info())

        self.energy_label = QLabel("Total energy: 50.35 GJ")
        self.energy_label.setStyleSheet("color: #1565C0; font-weight: bold;")
        mass_form.addRow("", self.energy_label)

        mass_group.setLayout(mass_form)
        layout.addWidget(mass_group)

        # Efficiency
        eff_group = QGroupBox("⚡ Explosion Efficiency")
        eff_layout = QVBoxLayout()

        eff_slider_layout = QHBoxLayout()
        self.eff_slider = QSlider(Qt.Orientation.Horizontal)
        self.eff_slider.setRange(1, 10)
        self.eff_slider.setValue(4)
        self.eff_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.eff_slider.setTickInterval(1)
        self.eff_slider.valueChanged.connect(self._on_eff_slider)
        eff_slider_layout.addWidget(QLabel("1%"))
        eff_slider_layout.addWidget(self.eff_slider)
        eff_slider_layout.addWidget(QLabel("10%"))
        eff_layout.addLayout(eff_slider_layout)

        eff_spin_layout = QHBoxLayout()
        eff_spin_layout.addWidget(QLabel("Efficiency η:"))
        self.eff_spin = QDoubleSpinBox()
        self.eff_spin.setRange(0.01, 0.10)
        self.eff_spin.setValue(0.04)
        self.eff_spin.setDecimals(2)
        self.eff_spin.setSingleStep(0.01)
        self.eff_spin.valueChanged.connect(self._on_eff_spin)
        eff_spin_layout.addWidget(self.eff_spin)
        eff_spin_layout.addStretch()
        eff_layout.addLayout(eff_spin_layout)

        self.tnt_equiv_label = QLabel("TNT equivalent: — kg")
        self.tnt_equiv_label.setStyleSheet("color: #E65100; font-weight: bold;")
        eff_layout.addWidget(self.tnt_equiv_label)

        eff_group.setLayout(eff_layout)
        layout.addWidget(eff_group)

        self._update_info()
        layout.addStretch()

    def _on_eff_slider(self, value: int):
        self.eff_spin.setValue(value / 100.0)

    def _on_eff_spin(self, value: float):
        self.eff_slider.blockSignals(True)
        self.eff_slider.setValue(int(value * 100))
        self.eff_slider.blockSignals(False)
        self._update_info()

    def _update_info(self):
        mass = self.mass_spin.value()
        delta_hc = self.substance_group.get_delta_hc()
        energy = mass * delta_hc
        if energy >= 1e9:
            self.energy_label.setText(f"Total energy: {energy/1e9:.2f} GJ")
        elif energy >= 1e6:
            self.energy_label.setText(f"Total energy: {energy/1e6:.2f} MJ")
        else:
            self.energy_label.setText(f"Total energy: {energy/1e3:.2f} kJ")

        eff = self.eff_spin.value()
        tnt_equiv = eff * mass * delta_hc / 4.68e6
        self.tnt_equiv_label.setText(f"TNT equivalent: {tnt_equiv:.1f} kg")

    def get_params(self) -> Dict[str, Any]:
        return {
            "mass_flammable": self.mass_spin.value(),
            "heat_of_combustion": self.substance_group.get_delta_hc(),
            "explosion_efficiency": self.eff_spin.value(),
            "substance_name": self.substance_group.get_substance_name(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TNO Multi-Energy Tab
# ══════════════════════════════════════════════════════════════════════════════

class TNOInputTab(QWidget):
    """TNO Multi-Energy input parameters."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()
        self._updating = False

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Substance selector
        self.substance_group = SubstanceSelectorGroup()
        layout.addWidget(self.substance_group)

        # Mass
        mass_group = QGroupBox("📦 Release Parameters")
        mass_form = QFormLayout()

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.1, 1e9)
        self.mass_spin.setValue(1000.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.valueChanged.connect(self._on_params_changed)
        mass_form.addRow("Mass Flammable:", self.mass_spin)

        self.energy_label = QLabel("Total energy: 50.35 GJ")
        self.energy_label.setStyleSheet("color: #1565C0; font-weight: bold;")
        mass_form.addRow("", self.energy_label)

        mass_group.setLayout(mass_form)
        layout.addWidget(mass_group)

        # Confinement
        conf_group = QGroupBox("🏗️ Confinement & Congestion")
        conf_form = QFormLayout()

        self.confinement_combo = QComboBox()
        for c in TNO_CONFINEMENT_CLASSES:
            desc = CONFINEMENT_DESC.get(c, "")
            self.confinement_combo.addItem(f"{c} — {desc}", c)
        self.confinement_combo.setCurrentIndex(2)  # default "1D"
        self.confinement_combo.currentIndexChanged.connect(self._on_params_changed)
        conf_form.addRow("Confinement:", self.confinement_combo)

        self.congestion_combo = QComboBox()
        for c in CONGESTION_LEVELS:
            desc = CONGESTION_DESC.get(c, "")
            self.congestion_combo.addItem(f"{c} — {desc}", c)
        self.congestion_combo.setCurrentIndex(1)  # default "medium"
        self.congestion_combo.currentIndexChanged.connect(self._on_params_changed)
        conf_form.addRow("Congestion:", self.congestion_combo)

        conf_group.setLayout(conf_form)
        layout.addWidget(conf_group)

        # Blast Strength
        blast_group = QGroupBox("💪 Blast Strength")
        blast_layout = QVBoxLayout()

        self.auto_strength_check = QCheckBox("Auto-select from confinement/congestion")
        self.auto_strength_check.setChecked(True)
        self.auto_strength_check.toggled.connect(self._on_auto_toggled)
        blast_layout.addWidget(self.auto_strength_check)

        strength_slider_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(1, 10)
        self.strength_slider.setValue(5)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setTickInterval(1)
        self.strength_slider.setEnabled(False)
        self.strength_slider.valueChanged.connect(self._on_strength_slider)
        strength_slider_layout.addWidget(QLabel("1 (Weak)"))
        strength_slider_layout.addWidget(self.strength_slider)
        strength_slider_layout.addWidget(QLabel("10 (Max)"))
        blast_layout.addLayout(strength_slider_layout)

        self.strength_label = QLabel("Blast Strength: 5 — Moderate")
        self.strength_label.setStyleSheet("font-weight: bold; color: #E65100;")
        blast_layout.addWidget(self.strength_label)

        self.strength_desc_label = QLabel("")
        self.strength_desc_label.setStyleSheet("color: #555; font-style: italic;")
        self.strength_desc_label.setWordWrap(True)
        blast_layout.addWidget(self.strength_desc_label)

        blast_group.setLayout(blast_layout)
        layout.addWidget(blast_group)

        self._update_auto_strength()
        self._on_params_changed()
        layout.addStretch()

    def _on_auto_toggled(self, checked: bool):
        self.strength_slider.setEnabled(not checked)
        if checked:
            self._update_auto_strength()
        self._on_strength_changed()

    def _on_params_changed(self):
        if self.auto_strength_check.isChecked():
            self._update_auto_strength()
        self._update_energy()

    def _on_strength_slider(self, value: int):
        if not self._updating:
            self._on_strength_changed()

    def _update_auto_strength(self):
        """Calculate auto blast strength from confinement + congestion."""
        try:
            from rekarisk.models.explosion.tno_multi_energy import (
                auto_blast_strength, blast_strength_description,
            )
            conf = self.confinement_combo.currentData()
            cong = self.congestion_combo.currentData()
            strength = auto_blast_strength(conf, cong)

            self._updating = True
            self.strength_slider.setValue(strength)
            self._updating = False

            desc = blast_strength_description(strength)
            self.strength_label.setText(f"Blast Strength: {strength} — {desc[:20]}...")
            self.strength_desc_label.setText(desc)
        except ImportError:
            pass

    def _on_strength_changed(self):
        strength = self.strength_slider.value()
        try:
            from rekarisk.models.explosion.tno_multi_energy import (
                blast_strength_description,
            )
            desc = blast_strength_description(strength)
        except ImportError:
            desc = f"Strength {strength}"
        self.strength_label.setText(f"Blast Strength: {strength}")
        self.strength_desc_label.setText(desc)

    def _update_energy(self):
        mass = self.mass_spin.value()
        dhc = self.substance_group.get_delta_hc()
        energy = mass * dhc
        if energy >= 1e9:
            self.energy_label.setText(f"Total energy: {energy/1e9:.2f} GJ")
        else:
            self.energy_label.setText(f"Total energy: {energy/1e6:.2f} MJ")

    def get_params(self) -> Dict[str, Any]:
        return {
            "tno_mass_flammable": self.mass_spin.value(),
            "tno_heat_of_combustion": self.substance_group.get_delta_hc(),
            "tno_substance_name": self.substance_group.get_substance_name(),
            "confinement_class": self.confinement_combo.currentData(),
            "congestion_level": self.congestion_combo.currentData(),
            "blast_strength": self.strength_slider.value()
            if not self.auto_strength_check.isChecked() else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Baker-Strehlow-Tang Tab
# ══════════════════════════════════════════════════════════════════════════════

class BSTInputTab(QWidget):
    """Baker-Strehlow-Tang input parameters."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Substance selector (also provides reactivity auto-lookup)
        self.substance_group = SubstanceSelectorGroup()
        layout.addWidget(self.substance_group)

        # Mass
        mass_group = QGroupBox("📦 Release Parameters")
        mass_form = QFormLayout()

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.1, 1e9)
        self.mass_spin.setValue(1000.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.valueChanged.connect(self._update_energy)
        mass_form.addRow("Mass Flammable:", self.mass_spin)

        self.energy_label = QLabel("Total energy: 50.35 GJ")
        self.energy_label.setStyleSheet("color: #1565C0; font-weight: bold;")
        mass_form.addRow("", self.energy_label)

        mass_group.setLayout(mass_form)
        layout.addWidget(mass_group)

        # Fuel Reactivity
        reactivity_group = QGroupBox("🔥 Fuel Reactivity")
        react_form = QFormLayout()

        self.reactivity_combo = QComboBox()
        self.reactivity_combo.addItem("Auto (from substance)", None)
        for r in REACTIVITY_LEVELS:
            desc = REACTIVITY_DESC.get(r, "")
            self.reactivity_combo.addItem(f"{r} — {desc}", r)
        self.reactivity_combo.currentIndexChanged.connect(self._on_params_changed)
        react_form.addRow("Reactivity:", self.reactivity_combo)

        react_group = QFormLayout()
        react_group.addRow(self.reactivity_combo)

        reactivity_group.setLayout(react_form)
        layout.addWidget(reactivity_group)

        # Confinement & Congestion
        conf_group = QGroupBox("🏗️ Confinement & Congestion")
        conf_form = QFormLayout()

        self.confinement_combo = QComboBox()
        for c in CONFINEMENT_CLASSES:
            desc = CONFINEMENT_DESC.get(c, "")
            self.confinement_combo.addItem(f"{c} — {desc}", c)
        self.confinement_combo.setCurrentIndex(0)
        self.confinement_combo.currentIndexChanged.connect(self._on_params_changed)
        conf_form.addRow("Confinement:", self.confinement_combo)

        self.congestion_combo = QComboBox()
        for c in CONGESTION_LEVELS:
            desc = CONGESTION_DESC.get(c, "")
            self.congestion_combo.addItem(f"{c} — {desc}", c)
        self.congestion_combo.setCurrentIndex(1)
        self.congestion_combo.currentIndexChanged.connect(self._on_params_changed)
        conf_form.addRow("Congestion:", self.congestion_combo)

        conf_group.setLayout(conf_form)
        layout.addWidget(conf_group)

        # Mach Number (calculated)
        mach_group = QGroupBox("⚡ Flame Speed (Mach Number)")
        mach_layout = QFormLayout()

        self.auto_mach_check = QCheckBox("Auto-calculate from reactivity/confinement/congestion")
        self.auto_mach_check.setChecked(True)
        self.auto_mach_check.toggled.connect(self._on_params_changed)
        mach_layout.addRow("", self.auto_mach_check)

        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(0.05, 5.0)
        self.mach_spin.setValue(0.5)
        self.mach_spin.setDecimals(2)
        self.mach_spin.setSingleStep(0.1)
        self.mach_spin.setEnabled(False)
        self.mach_spin.valueChanged.connect(self._on_mach_changed)
        mach_layout.addRow("Mach Number:", self.mach_spin)

        self.mach_desc_label = QLabel("Ma = 0.50 — Medium flame speed")
        self.mach_desc_label.setStyleSheet("color: #E65100; font-weight: bold;")
        self.mach_desc_label.setWordWrap(True)
        mach_layout.addRow("", self.mach_desc_label)

        mach_group.setLayout(mach_layout)
        layout.addWidget(mach_group)

        self._on_params_changed()
        self._update_energy()
        layout.addStretch()

    def _on_params_changed(self):
        self.mach_spin.setEnabled(not self.auto_mach_check.isChecked())
        if self.auto_mach_check.isChecked():
            self._update_auto_mach()

    def _on_mach_changed(self, value):
        self._update_mach_label(value)

    def _update_auto_mach(self):
        """Calculate Mach number from reactivity + confinement + congestion."""
        try:
            from rekarisk.models.explosion.baker_strehlow import (
                mach_from_confinement_congestion,
            )
            react = self._get_effective_reactivity()
            conf = self.confinement_combo.currentData()
            cong = self.congestion_combo.currentData()
            mach = mach_from_confinement_congestion(conf, cong, react)
            self.mach_spin.blockSignals(True)
            self.mach_spin.setValue(mach)
            self.mach_spin.blockSignals(False)
            self._update_mach_label(mach)
        except ImportError:
            pass

    def _get_effective_reactivity(self) -> str:
        """Get reactivity: manual or auto from substance."""
        react_data = self.reactivity_combo.currentData()
        if react_data:
            return react_data
        return self.substance_group.get_reactivity()

    def _update_mach_label(self, mach: float):
        if mach < 0.3:
            desc = "Low flame speed — minimal blast"
        elif mach < 0.7:
            desc = "Moderate-low flame speed"
        elif mach < 1.0:
            desc = "Moderate flame speed — subsonic"
        elif mach < 1.5:
            desc = "High subsonic flame speed"
        elif mach < 3.0:
            desc = "Supersonic flame — strong blast"
        else:
            desc = "DDT / detonation — maximum blast"
        self.mach_desc_label.setText(f"Ma = {mach:.2f} — {desc}")

    def _update_energy(self):
        mass = self.mass_spin.value()
        dhc = self.substance_group.get_delta_hc()
        energy = mass * dhc
        if energy >= 1e9:
            self.energy_label.setText(f"Total energy: {energy/1e9:.2f} GJ")
        else:
            self.energy_label.setText(f"Total energy: {energy/1e6:.2f} MJ")

    def get_params(self) -> Dict[str, Any]:
        react_data = self.reactivity_combo.currentData()
        return {
            "bst_mass_flammable": self.mass_spin.value(),
            "bst_heat_of_combustion": self.substance_group.get_delta_hc(),
            "bst_substance_name": self.substance_group.get_substance_name(),
            "fuel_reactivity": react_data,
            "bst_confinement_class": self.confinement_combo.currentData(),
            "bst_congestion_level": self.congestion_combo.currentData(),
            "flame_mach": self.mach_spin.value()
            if not self.auto_mach_check.isChecked() else None,
        }
