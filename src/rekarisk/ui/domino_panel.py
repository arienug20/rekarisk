"""
Rekarisk UI — Domino / Escalation Analysis Input Panel.

Provides a PyQt6 panel for configuring and running domino effect analysis:
  - Equipment table (add/remove/edit equipment items)
  - Primary event configuration (fire type, frequency, thermal power, TNT mass)
  - Analysis settings (max escalation order, response time, vectors)
  - Run button

Signals:
    calculation_requested(dict): Emitted when Run is clicked with full params.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QFormLayout, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QScrollArea, QMessageBox, QToolBar,
    QAbstractItemView, QStatusBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

EQUIPMENT_TYPES = [
    ("Atmospheric Tank", "atmospheric_tank"),
    ("Pressure Vessel", "pressure_vessel"),
    ("Reactor", "reactor"),
    ("Heat Exchanger", "heat_exchanger"),
    ("Pipeline", "pipeline"),
    ("Column / Tower", "column"),
    ("Separator", "separator"),
    ("Pump", "pump"),
    ("Compressor", "compressor"),
    ("Fin-Fan Cooler", "fin_fan_cooler"),
    ("Structure", "structure"),
]

SUBSTANCE_CATEGORIES = [
    ("Flammable Liquid", "flammable_liquid"),
    ("Flammable Gas", "flammable_gas"),
    ("Flammable LPG", "flammable_lpg"),
    ("Toxic", "toxic"),
    ("Reactive", "reactive"),
    ("Inert", "inert"),
]

EVENT_TYPES = [
    ("Pool Fire", "pool_fire"),
    ("Jet Fire", "jet_fire"),
    ("BLEVE", "bleve"),
    ("VCE", "vce"),
    ("Flash Fire", "flash_fire"),
]

# Column indices for equipment table
COL_ID = 0
COL_NAME = 1
COL_TYPE = 2
COL_SUBSTANCE = 3
COL_CATEGORY = 4
COL_INVENTORY = 5
COL_X = 6
COL_Y = 7
COL_DIAMETER = 8
COL_HEIGHT = 9
COL_PRESSURE = 10
COL_INSULATED = 11
COL_DELUGE = 12

TABLE_HEADERS = [
    "ID", "Name", "Type", "Substance", "Category",
    "Inventory (kg)", "X (m)", "Y (m)",
    "Dia (m)", "Height (m)", "P_op (bar)",
    "Insulated", "Deluge",
]


# ══════════════════════════════════════════════════════════════════════════════
# Domino Analysis Panel
# ══════════════════════════════════════════════════════════════════════════════

class DominoPanel(QWidget):
    """Domino / Escalation analysis input panel.

    Layout:
      - Top: Primary event configuration
      - Middle: Equipment table (editable)
      - Bottom: Analysis settings + Run button

    Signals:
        calculation_requested(dict): Full analysis parameters.
    """

    calculation_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Primary Event Group ──
        primary_group = QGroupBox("🔥 Primary Event")
        primary_form = QFormLayout(primary_group)
        primary_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._event_type_combo = QComboBox()
        for label, value in EVENT_TYPES:
            self._event_type_combo.addItem(label, value)
        primary_form.addRow("Event Type:", self._event_type_combo)

        self._primary_equipment_combo = QComboBox()
        self._primary_equipment_combo.setMinimumWidth(200)
        primary_form.addRow("Primary Equipment:", self._primary_equipment_combo)

        self._frequency_spin = QDoubleSpinBox()
        self._frequency_spin.setRange(1e-10, 1e-1)
        self._frequency_spin.setDecimals(8)
        self._frequency_spin.setValue(5e-6)
        self._frequency_spin.setPrefix("f = ")
        self._frequency_spin.setSuffix(" /yr")
        primary_form.addRow("Frequency:", self._frequency_spin)

        self._thermal_power_spin = QDoubleSpinBox()
        self._thermal_power_spin.setRange(0, 1e8)
        self._thermal_power_spin.setDecimals(0)
        self._thermal_power_spin.setValue(100000)
        self._thermal_power_spin.setSuffix(" kW")
        self._thermal_power_spin.setSingleStep(10000)
        primary_form.addRow("Thermal Power:", self._thermal_power_spin)

        self._tnt_mass_spin = QDoubleSpinBox()
        self._tnt_mass_spin.setRange(0, 1e6)
        self._tnt_mass_spin.setDecimals(1)
        self._tnt_mass_spin.setValue(100)
        self._tnt_mass_spin.setSuffix(" kg TNT")
        primary_form.addRow("TNT Equivalent:", self._tnt_mass_spin)

        self._fireball_spin = QDoubleSpinBox()
        self._fireball_spin.setRange(0, 500)
        self._fireball_spin.setDecimals(1)
        self._fireball_spin.setValue(30)
        self._fireball_spin.setSuffix(" m")
        primary_form.addRow("Fireball Radius:", self._fireball_spin)

        self._source_height_spin = QDoubleSpinBox()
        self._source_height_spin.setRange(0, 100)
        self._source_height_spin.setDecimals(1)
        self._source_height_spin.setValue(4.0)
        self._source_height_spin.setSuffix(" m")
        primary_form.addRow("Source Height:", self._source_height_spin)

        self._pool_radius_spin = QDoubleSpinBox()
        self._pool_radius_spin.setRange(0, 200)
        self._pool_radius_spin.setDecimals(1)
        self._pool_radius_spin.setValue(10)
        self._pool_radius_spin.setSuffix(" m")
        primary_form.addRow("Pool Radius:", self._pool_radius_spin)

        layout.addWidget(primary_group)

        # ── Equipment Table Group ──
        equip_group = QGroupBox("🏭 Equipment Layout")
        equip_layout = QVBoxLayout(equip_group)

        # Toolbar
        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("➕ Add Equipment")
        self._add_btn.clicked.connect(self._add_equipment_row)
        self._remove_btn = QPushButton("🗑 Remove Selected")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_table)
        toolbar.addWidget(self._add_btn)
        toolbar.addWidget(self._remove_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addStretch()
        equip_layout.addLayout(toolbar)

        # Table
        self._table = QTableWidget(0, len(TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(250)
        equip_layout.addWidget(self._table)

        # Pre-populate with demo data
        self._load_demo_data()

        layout.addWidget(equip_group)

        # ── Analysis Settings Group ──
        settings_group = QGroupBox("⚙️ Analysis Settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._max_order_spin = QSpinBox()
        self._max_order_spin.setRange(2, 5)
        self._max_order_spin.setValue(3)
        settings_form.addRow("Max Escalation Order:", self._max_order_spin)

        self._response_time_spin = QDoubleSpinBox()
        self._response_time_spin.setRange(1, 120)
        self._response_time_spin.setDecimals(1)
        self._response_time_spin.setValue(10.0)
        self._response_time_spin.setSuffix(" min")
        settings_form.addRow("Response Time:", self._response_time_spin)

        # Escalation vectors
        vec_layout = QHBoxLayout()
        self._thermal_check = QCheckBox("Thermal Radiation")
        self._thermal_check.setChecked(True)
        self._overpressure_check = QCheckBox("Overpressure")
        self._overpressure_check.setChecked(True)
        self._impingement_check = QCheckBox("Fire Impingement")
        self._impingement_check.setChecked(True)
        vec_layout.addWidget(self._thermal_check)
        vec_layout.addWidget(self._overpressure_check)
        vec_layout.addWidget(self._impingement_check)
        settings_form.addRow("Escalation Vectors:", vec_layout)

        layout.addWidget(settings_group)

        # ── Run Button ──
        run_layout = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Domino Analysis")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; "
            "background-color: #2e86c1; color: white; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1a5276; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        run_layout.addStretch()
        run_layout.addWidget(self._run_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Equipment table helpers ──

    def _add_equipment_row(
        self,
        eq_id: str = "",
        name: str = "",
        eq_type: str = "atmospheric_tank",
        substance: str = "",
        category: str = "flammable_liquid",
        inventory: float = 0,
        x: float = 0, y: float = 0,
        diameter: float = 2, height: float = 5,
        pressure: float = 1,
        insulated: bool = False,
        deluge: bool = False,
    ):
        row = self._table.rowCount()
        self._table.insertRow(row)

        # ID
        self._table.setItem(row, COL_ID, QTableWidgetItem(eq_id))
        # Name
        self._table.setItem(row, COL_NAME, QTableWidgetItem(name))

        # Type combo
        type_combo = QComboBox()
        for label, val in EQUIPMENT_TYPES:
            type_combo.addItem(label, val)
        idx = type_combo.findData(eq_type)
        if idx >= 0:
            type_combo.setCurrentIndex(idx)
        self._table.setCellWidget(row, COL_TYPE, type_combo)

        # Substance
        self._table.setItem(row, COL_SUBSTANCE, QTableWidgetItem(substance))

        # Category combo
        cat_combo = QComboBox()
        for label, val in SUBSTANCE_CATEGORIES:
            cat_combo.addItem(label, val)
        idx = cat_combo.findData(category)
        if idx >= 0:
            cat_combo.setCurrentIndex(idx)
        self._table.setCellWidget(row, COL_CATEGORY, cat_combo)

        # Numeric fields
        for col, val, decimals in [
            (COL_INVENTORY, inventory, 0),
            (COL_X, x, 1),
            (COL_Y, y, 1),
            (COL_DIAMETER, diameter, 1),
            (COL_HEIGHT, height, 1),
            (COL_PRESSURE, pressure, 1),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e8)
            spin.setDecimals(decimals)
            spin.setValue(val)
            self._table.setCellWidget(row, col, spin)

        # Checkboxes
        for col, checked in [(COL_INSULATED, insulated), (COL_DELUGE, deluge)]:
            chk = QCheckBox()
            chk.setChecked(checked)
            chk_container = QWidget()
            chk_layout = QHBoxLayout(chk_container)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, col, chk_container)

        self._update_primary_combo()

    def _remove_selected(self):
        rows = set(idx.row() for idx in self._table.selectedIndexes())
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)
        self._update_primary_combo()

    def _clear_table(self):
        self._table.setRowCount(0)
        self._update_primary_combo()

    def _update_primary_combo(self):
        """Update the primary equipment combo from table."""
        current = self._primary_equipment_combo.currentData()
        self._primary_equipment_combo.clear()
        for row in range(self._table.rowCount()):
            eq_id = self._table.item(row, COL_ID)
            if eq_id:
                self._primary_equipment_combo.addItem(eq_id.text(), eq_id.text())
        # Restore selection
        idx = self._primary_equipment_combo.findData(current)
        if idx >= 0:
            self._primary_equipment_combo.setCurrentIndex(idx)
        elif self._primary_equipment_combo.count() > 0:
            self._primary_equipment_combo.setCurrentIndex(0)

    def _load_demo_data(self):
        """Load demo propane storage area layout."""
        demo = [
            ("TK-301", "Propane Tank", "atmospheric_tank", "Propane", "flammable_lpg",
             50000, 0, 0, 10, 8, 8, False, False),
            ("V-302", "Separator", "pressure_vessel", "Propane", "flammable_lpg",
             5000, 30, 10, 2, 6, 12, True, False),
            ("V-303", "Butane Vessel", "pressure_vessel", "Butane", "flammable_lpg",
             20000, 40, -15, 4, 7, 5, False, False),
            ("C-304", "De-propanizer", "column", "LPG", "flammable_lpg",
             8000, -20, 25, 2, 20, 15, False, True),
            ("TK-305", "Condensate Tank", "atmospheric_tank", "Condensate", "flammable_liquid",
             100000, -30, -20, 12, 10, 0.1, False, False),
            ("HX-306", "Chiller", "heat_exchanger", "Propane", "flammable_lpg",
             2000, 15, 25, 1, 3, 10, False, False),
            ("P-307A", "Pump A", "pump", "Propane", "flammable_lpg",
             500, 15, -5, 0.5, 1.5, 10, False, False),
            ("P-307B", "Pump B", "pump", "Propane", "flammable_lpg",
             500, 20, 5, 0.5, 1.5, 10, False, False),
            ("K-308", "Compressor", "compressor", "Propane", "flammable_gas",
             1000, -10, 15, 2, 3, 15, False, False),
            ("FIN-309", "Air Cooler", "fin_fan_cooler", "Propane", "flammable_lpg",
             1500, 5, -15, 3, 2, 12, False, False),
        ]
        for row_data in demo:
            self._add_equipment_row(*row_data)

    # ── Collect parameters ──

    def get_all_params(self) -> Dict[str, Any]:
        """Collect all parameters for the domino analysis."""
        equipment_list = []
        for row in range(self._table.rowCount()):
            eq_id = self._table.item(row, COL_ID)
            name = self._table.item(row, COL_NAME)
            type_combo = self._table.cellWidget(row, COL_TYPE)
            substance = self._table.item(row, COL_SUBSTANCE)
            cat_combo = self._table.cellWidget(row, COL_CATEGORY)

            if not eq_id or not eq_id.text().strip():
                continue

            # Get numeric values
            inventory = self._table.cellWidget(row, COL_INVENTORY)
            x_spin = self._table.cellWidget(row, COL_X)
            y_spin = self._table.cellWidget(row, COL_Y)
            dia_spin = self._table.cellWidget(row, COL_DIAMETER)
            h_spin = self._table.cellWidget(row, COL_HEIGHT)
            p_spin = self._table.cellWidget(row, COL_PRESSURE)

            # Get checkboxes
            ins_widget = self._table.cellWidget(row, COL_INSULATED)
            del_widget = self._table.cellWidget(row, COL_DELUGE)
            insulated = ins_widget.findChild(QCheckBox).isChecked() if ins_widget else False
            has_deluge = del_widget.findChild(QCheckBox).isChecked() if del_widget else False

            equip = {
                "id": eq_id.text().strip(),
                "name": name.text().strip() if name else eq_id.text().strip(),
                "equipment_type": type_combo.currentData() if type_combo else "pressure_vessel",
                "substance": substance.text().strip() if substance else "Unknown",
                "substance_category": cat_combo.currentData() if cat_combo else "flammable_liquid",
                "inventory_kg": inventory.value() if inventory else 0,
                "x": x_spin.value() if x_spin else 0,
                "y": y_spin.value() if y_spin else 0,
                "diameter": dia_spin.value() if dia_spin else 2,
                "height": h_spin.value() if h_spin else 5,
                "operating_pressure": p_spin.value() if p_spin else 1,
                "is_insulated": insulated,
                "has_deluge": has_deluge,
            }
            equipment_list.append(equip)

        return {
            "primary_equipment_id": self._primary_equipment_combo.currentData() or "",
            "event_type": self._event_type_combo.currentData(),
            "frequency": self._frequency_spin.value(),
            "thermal_power_kw": self._thermal_power_spin.value(),
            "tnt_mass_kg": self._tnt_mass_spin.value(),
            "fireball_radius_m": self._fireball_spin.value(),
            "source_height_m": self._source_height_spin.value(),
            "pool_radius_m": self._pool_radius_spin.value(),
            "max_escalation_order": self._max_order_spin.value(),
            "response_time_min": self._response_time_spin.value(),
            "include_thermal": self._thermal_check.isChecked(),
            "include_overpressure": self._overpressure_check.isChecked(),
            "include_impingement": self._impingement_check.isChecked(),
            "equipment": equipment_list,
        }

    def _on_run(self):
        """Collect parameters and emit calculation_requested."""
        params = self.get_all_params()
        if not params.get("equipment"):
            QMessageBox.warning(self, "No Equipment", "Add at least one equipment item.")
            return
        if not params.get("primary_equipment_id"):
            QMessageBox.warning(self, "No Primary", "Select a primary equipment for the initiating event.")
            return
        self.calculation_requested.emit(params)
