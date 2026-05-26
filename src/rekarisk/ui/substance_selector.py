"""
Rekarisk UI — Substance Selector.

Searchable substance picker widget with autocomplete, property preview,
and filtering by phase, hazard class, or tag.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QComboBox, QGroupBox, QFormLayout,
    QSplitter, QFrame, QTextEdit,
)

from ..core.substance import Substance
from ..core.substance_db import SubstanceDatabase, get_database


class SubstanceSelector(QWidget):
    """Searchable substance selector with property preview.

    Signals:
        substance_selected(Substance): Emitted when user selects a substance.
    """

    substance_selected = pyqtSignal(object)  # Substance

    def __init__(self, db: SubstanceDatabase | None = None, parent=None):
        super().__init__(parent)
        self._db = db or get_database()
        self._all_substances: list[Substance] = list(self._db)
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)  # debounce 300ms
        self._search_timer.timeout.connect(self._do_search)

        self._setup_ui()
        self._populate_list(self._all_substances)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ── Search bar ──
        search_layout = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search substances (name, CAS, formula)...")
        self._search_input.textChanged.connect(self._on_search_text_changed)
        search_layout.addWidget(self._search_input, 3)

        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All", None)
        self._filter_combo.addItem("Flammable", "flammable")
        self._filter_combo.addItem("Toxic", "toxic")
        self._filter_combo.addItem("Gases", "gas")
        self._filter_combo.addItem("Liquids", "liquid")
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        search_layout.addWidget(self._filter_combo, 1)

        main_layout.addLayout(search_layout)

        # ── Splitter: list + detail ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Substance list
        self._list_widget = QListWidget()
        self._list_widget.setAlternatingRowColors(True)
        self._list_widget.currentItemChanged.connect(self._on_list_selection)
        splitter.addWidget(self._list_widget)

        # Property preview
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 4, 0, 0)

        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(200)
        self._detail_text.setStyleSheet(
            "QTextEdit { font-size: 12px; background: #f5f5f5; "
            "border: 1px solid #cccccc; border-radius: 4px; }"
        )
        detail_layout.addWidget(QLabel("Properties:"))
        detail_layout.addWidget(self._detail_text)

        splitter.addWidget(detail_widget)
        splitter.setSizes([300, 150])

        main_layout.addWidget(splitter)

    # ── Public API ──

    def set_database(self, db: SubstanceDatabase):
        """Replace the database source."""
        self._db = db
        self._all_substances = list(db)
        self.refresh()

    def refresh(self):
        """Reload the substance list from the database."""
        current_filter = self._filter_combo.currentData()
        self._all_substances = list(self._db)
        if current_filter:
            if current_filter in ("flammable", "toxic"):
                self._all_substances = self._db.filter_by_hazard(current_filter)
            elif current_filter in ("gas", "liquid"):
                self._all_substances = self._db.filter_by_phase(current_filter)
        self._populate_list(self._all_substances)

    def get_selected(self) -> Substance | None:
        """Return the currently selected substance."""
        item = self._list_widget.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    # ── Internal ──

    def _populate_list(self, substances: list[Substance]):
        self._list_widget.clear()
        for sub in substances:
            # Format: "Name  (CAS: xxx)  [phase]"
            phase = "G" if sub.is_gas_at_ambient else "L"
            display = f"{sub.name}   [{phase}]"
            if sub.cas_number:
                display += f"   CAS: {sub.cas_number}"
            if sub.formula:
                display += f"   {sub.formula}"

            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, sub)
            item.setToolTip(
                f"MW: {sub.molecular_weight:.1f} g/mol  "
                f"FP: {sub.flash_point - 273.15:.0f}°C" if sub.flash_point else ""
            )
            self._list_widget.addItem(item)

        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)

    def _show_property_preview(self, sub: Substance):
        """Display key properties of the selected substance."""
        lines = []
        lines.append(f"<b>{sub.name}</b>")
        if sub.formula:
            lines.append(f"Formula: {sub.formula}")
        if sub.cas_number:
            lines.append(f"CAS: {sub.cas_number}")
        if sub.un_number:
            lines.append(f"UN: {sub.un_number}")
        lines.append("")

        lines.append(f"Molecular Weight: {sub.molecular_weight:.2f} g/mol")

        if sub.normal_boiling_point is not None:
            bp = sub.normal_boiling_point - 273.15
            lines.append(f"Boiling Point: {bp:.1f} °C")

        if sub.flash_point is not None:
            fp = sub.flash_point - 273.15
            lines.append(f"Flash Point: {fp:.1f} °C")

        if sub.auto_ignition_temp is not None:
            ait = sub.auto_ignition_temp - 273.15
            lines.append(f"Auto-ignition Temp: {ait:.1f} °C")

        if sub.lower_flammability_limit is not None:
            lines.append(f"LFL/UFL: {sub.lower_flammability_limit:.2f} / "
                         f"{sub.upper_flammability_limit:.2f} (vol)")

        if sub.liquid_density is not None:
            lines.append(f"Liquid Density: {sub.liquid_density:.1f} kg/m³")

        if sub.heat_of_combustion is not None:
            hoc_mj = sub.heat_of_combustion / 1e6
            lines.append(f"Heat of Combustion: {hoc_mj:.1f} MJ/kg")

        if sub.erpg2 is not None:
            lines.append(f"ERPG-2: {sub.erpg2:.1f} mg/m³")

        if sub.idlh is not None:
            lines.append(f"IDLH: {sub.idlh:.0f} ppm")

        phase = "Gas" if sub.is_gas_at_ambient else "Liquid"
        lines.append(f"")
        lines.append(f"Phase at ambient: <b>{phase}</b>")
        lines.append(f"Hazard classes: {', '.join(sub.hazard_classes) or 'none'}")

        self._detail_text.setHtml("<br>".join(lines))

    # ── Slots ──

    def _on_search_text_changed(self, text: str):
        self._search_timer.start()  # debounce

    def _do_search(self):
        query = self._search_input.text().strip()
        if not query:
            results = self._all_substances
        else:
            results = self._db.search(query, max_results=50)
            # Also apply current filter
            current_filter = self._filter_combo.currentData()
            if current_filter:
                if current_filter in ("flammable", "toxic"):
                    results = [s for s in results if current_filter in s.hazard_classes]
                elif current_filter in ("gas", "liquid"):
                    results = [s for s in results
                               if s.phase_at_ambient == current_filter]
        self._populate_list(results)

    def _on_filter_changed(self, index: int):
        filter_val = self._filter_combo.currentData()
        if filter_val is None:
            self._all_substances = list(self._db)
        elif filter_val in ("flammable", "toxic"):
            self._all_substances = self._db.filter_by_hazard(filter_val)
        elif filter_val in ("gas", "liquid"):
            self._all_substances = self._db.filter_by_phase(filter_val)
        self._do_search()

    def _on_list_selection(self, current, previous):
        if current:
            sub = current.data(Qt.ItemDataRole.UserRole)
            if sub:
                self._show_property_preview(sub)
                self.substance_selected.emit(sub)
