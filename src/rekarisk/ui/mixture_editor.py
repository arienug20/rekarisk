"""
Rekarisk — Mixture Composition Editor.

PyQt6 dialog for editing multi-component mixture compositions.
Supports mole/mass fraction entry, auto-normalization, property preview,
and integration with substance database + EoS engine.

Usage:
    from rekarisk.ui.mixture_editor import MixtureEditorDialog
    dialog = MixtureEditorDialog(parent, substance_db)
    if dialog.exec_():
        mole_fracs = dialog.mole_fractions
        components = dialog.components  # list of Substance
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QLineEdit, QHeaderView, QMessageBox,
    QGroupBox, QFormLayout, QAbstractItemView, QComboBox,
    QSplitter, QTextEdit, QDialogButtonBox, QWidget,
)
from PyQt6.QtGui import QDoubleValidator, QColor, QBrush


# Try to import substance database and EoS
try:
    from ..core.substance_db import SubstanceDatabase
    from ..core.substance import Substance
    from ..core.eos import create_mixture_eos, EoSParameters
    _HAS_SUBSTANCE_DB = True
except ImportError:
    _HAS_SUBSTANCE_DB = False
    Substance = None
    SubstanceDatabase = None
    EoSParameters = None
    create_mixture_eos = None

try:
    from .substance_selector import SubstanceSelector
    _HAS_SELECTOR = True
except ImportError:
    _HAS_SELECTOR = False


# ══════════════════════════════════════════════════════════════════════════════
# Mixture Editor Dialog
# ══════════════════════════════════════════════════════════════════════════════

class MixtureEditorDialog(QDialog):
    """Dialog for editing mixture compositions.

    Features:
      - Add/remove components via SubstanceSelector or quick search
      - Mole fraction or mass fraction entry
      - Auto-normalization (sum → 1.0)
      - Mixture property preview (MW, pseudo-critical T/P, acentric)
      - Validation before accept

    Signals:
        compositionChanged: emitted when fractions change (for live preview).
    """

    compositionChanged = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        substance_db: Optional["SubstanceDatabase"] = None,
        initial_components: Optional[List["Substance"]] = None,
        initial_fractions: Optional[List[float]] = None,
        mode: str = "mole",
    ):
        """Initialize mixture editor.

        Args:
            parent: Parent widget.
            substance_db: SubstanceDatabase instance for component lookup.
            initial_components: Pre-populate with these substances.
            initial_fractions: Pre-populate with these fractions.
            mode: 'mole' (default) or 'mass'.
        """
        super().__init__(parent)
        self.setWindowTitle("Mixture Composition Editor")
        self.setMinimumSize(800, 500)
        self.resize(900, 600)

        self._substance_db = substance_db
        self._mode = mode  # 'mole' or 'mass'
        self._components: List["Substance"] = []
        self._fractions: List[float] = []

        self._build_ui()

        # Load initial data if provided
        if initial_components and initial_fractions:
            self.load_mixture(initial_components, initial_fractions)

    # ══ UI Construction ══

    def _build_ui(self):
        """Build the complete UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Mode selector ──
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Fraction mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Mole Fraction", "Mass Fraction"])
        self._mode_combo.setCurrentIndex(0 if self._mode == "mole" else 1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── Splitter: table + preview ──
        splitter = QSplitter(Qt.Vertical)

        # Table widget
        table_group = QGroupBox("Components")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Component", "CAS", "Fraction", "Mass Frac", ""
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.setColumnWidth(2, 100)
        self._table.setColumnWidth(3, 100)
        table_layout.addWidget(self._table)

        # Button row
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("➕ Add Component")
        self._add_btn.clicked.connect(self._add_component)
        btn_row.addWidget(self._add_btn)

        self._remove_btn = QPushButton("➖ Remove Selected")
        self._remove_btn.clicked.connect(self._remove_component)
        btn_row.addWidget(self._remove_btn)

        self._normalize_btn = QPushButton("🔄 Normalize")
        self._normalize_btn.clicked.connect(self._normalize)
        self._normalize_btn.setToolTip("Scale all fractions so they sum to 1.0")
        btn_row.addWidget(self._normalize_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._clear_btn)

        btn_row.addStretch()

        # Load from DB
        self._load_db_btn = QPushButton("📋 Load from DB")
        self._load_db_btn.clicked.connect(self._load_from_db)
        btn_row.addWidget(self._load_db_btn)

        table_layout.addLayout(btn_row)
        splitter.addWidget(table_group)

        # ── Preview panel ──
        preview_group = QGroupBox("Mixture Properties Preview")
        preview_layout = QFormLayout(preview_group)

        self._mw_label = QLabel("—")
        self._tc_label = QLabel("—")
        self._pc_label = QLabel("—")
        self._omega_label = QLabel("—")
        self._sum_label = QLabel("—")
        self._count_label = QLabel("0")

        preview_layout.addRow("Number of components:", self._count_label)
        preview_layout.addRow("Sum of fractions:", self._sum_label)
        preview_layout.addRow("Mixture MW [g/mol]:", self._mw_label)
        preview_layout.addRow("Pseudo-critical Tc [K]:", self._tc_label)
        preview_layout.addRow("Pseudo-critical Pc [MPa]:", self._pc_label)
        preview_layout.addRow("Pseudo-acentric ω:", self._omega_label)

        splitter.addWidget(preview_group)
        layout.addWidget(splitter, stretch=1)

        # ── Messages area ──
        self._msg_text = QTextEdit()
        self._msg_text.setReadOnly(True)
        self._msg_text.setMaximumHeight(60)
        self._msg_text.setStyleSheet("QTextEdit { background: #f5f5f5; font-size: 12px; }")
        layout.addWidget(self._msg_text)

        # ── OK / Cancel ──
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ══ Slots ══

    def _on_mode_changed(self, index: int):
        """Toggle between mole fraction and mass fraction mode."""
        self._mode = "mole" if index == 0 else "mass"
        self._update_table_headers()
        # Convert existing fractions
        self._convert_fractions()
        self._refresh_table()
        self._update_preview()

    def _update_table_headers(self):
        """Update column headers based on mode."""
        col_names = ["Component", "CAS"]
        if self._mode == "mole":
            col_names += ["Mole Fraction", "Mass Frac", ""]
        else:
            col_names += ["Mass Fraction", "Mole Frac", ""]
        self._table.setHorizontalHeaderLabels(col_names)

    def _convert_fractions(self):
        """Convert fractions between mole and mass basis."""
        if not self._components:
            return
        total = sum(self._fractions)
        if total <= 0:
            return

        mws = np.array([c.molecular_weight for c in self._components])
        fracs = np.array(self._fractions) / total

        if self._mode == "mole":
            # Converting from mass to mole
            moles = fracs / mws
            moles /= moles.sum()
            self._fractions = moles.tolist()
        else:
            # Converting from mole to mass
            masses = fracs * mws
            masses /= masses.sum()
            self._fractions = masses.tolist()

    def _on_cell_changed(self, row: int, col: int):
        """Handle manual fraction edits."""
        if col != 2 or row >= len(self._components):
            return

        item = self._table.item(row, col)
        if item is None:
            return

        try:
            val = float(item.text())
            if val < 0 or val > 1.0:
                raise ValueError
            self._fractions[row] = val
        except ValueError:
            # Revert to previous value
            self._table.blockSignals(True)
            item.setText(f"{self._fractions[row]:.4f}")
            self._table.blockSignals(False)
            return

        self._update_preview()
        self.compositionChanged.emit()

    def _add_component(self):
        """Add a component via quick name/CAS input or SubstanceSelector."""
        if _HAS_SELECTOR and self._substance_db is not None:
            selector = SubstanceSelector(self, self._substance_db)
            if selector.exec_():
                substances = selector.selected_substances
                for sub in substances:
                    self._add_substance(sub)
        else:
            # Fallback: simple name input
            name, ok = self._simple_input("Component Name", "Enter component name or formula:")
            if ok and name:
                # Try to find in DB
                sub = None
                if self._substance_db:
                    results = self._substance_db.search(name)
                    if results:
                        sub = results[0]
                if sub is None and Substance:
                    sub = Substance(id=name, name=name, molecular_weight=16.0)
                if sub:
                    self._add_substance(sub)

    def _simple_input(self, title: str, prompt: str) -> tuple:
        """Simple text input dialog (fallback)."""
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, title, prompt)
        return text, ok

    def _add_substance(self, sub: "Substance"):
        """Add a substance to the table."""
        self._components.append(sub)
        self._fractions.append(0.0)
        self._refresh_table()
        self._msg(f"Added: {sub.name} (MW={sub.molecular_weight:.1f} g/mol)")

    def _remove_component(self):
        """Remove selected component(s)."""
        selected = set()
        for item in self._table.selectedItems():
            selected.add(item.row())
        if not selected:
            return

        for row in sorted(selected, reverse=True):
            name = self._components[row].name
            del self._components[row]
            del self._fractions[row]
            self._msg(f"Removed: {name}")

        self._refresh_table()
        self._update_preview()

    def _normalize(self):
        """Normalize fractions to sum = 1.0."""
        total = sum(self._fractions)
        if total <= 0:
            self._msg("⚠️  Cannot normalize: all fractions are zero.")
            return

        self._fractions = [f / total for f in self._fractions]
        self._refresh_table()
        self._update_preview()
        self._msg(f"✅ Normalized: sum = {sum(self._fractions):.10f}")

    def _clear_all(self):
        """Remove all components."""
        self._components.clear()
        self._fractions.clear()
        self._refresh_table()
        self._update_preview()
        self._msg("Cleared all components.")

    def _load_from_db(self):
        """Open substance selector to load from database."""
        if not _HAS_SELECTOR or self._substance_db is None:
            self._msg("⚠️  Substance database not available.")
            return

        selector = SubstanceSelector(self, self._substance_db,
                                     multi_select=True)
        if selector.exec_():
            for sub in selector.selected_substances:
                # Check not already in table
                if any(c.id == sub.id for c in self._components):
                    self._msg(f"⚠️  {sub.name} already in list.")
                    continue
                self._add_substance(sub)

    def _on_accept(self):
        """Validate and accept."""
        if len(self._components) == 0:
            QMessageBox.warning(self, "Validation",
                                "Please add at least one component.")
            return

        if len(self._components) == 1:
            self._fractions[0] = 1.0
            self._refresh_table()
            self._update_preview()

        total = sum(self._fractions)
        if abs(total - 1.0) > 0.001:
            reply = QMessageBox.question(
                self, "Normalize?",
                f"Fractions sum to {total:.4f}, not 1.0.\n\n"
                "Normalize automatically?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self._normalize()
            elif reply == QMessageBox.Cancel:
                return
            # If No, proceed with warning

        self.accept()

    # ══ Table Refresh ══

    def _refresh_table(self):
        """Rebuild the table from current _components and _fractions."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._components))

        mws = np.array([c.molecular_weight for c in self._components]) if self._components else np.array([])

        for i, (comp, frac) in enumerate(zip(self._components, self._fractions)):
            # Name
            name_item = QTableWidgetItem(comp.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, name_item)

            # CAS
            cas_item = QTableWidgetItem(comp.cas_number or "—")
            cas_item.setFlags(cas_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, cas_item)

            # Fraction (editable)
            frac_item = QTableWidgetItem(f"{frac:.6f}")
            self._table.setItem(i, 2, frac_item)

            # Other fraction (computed, not editable)
            if len(mws) == len(self._components) and len(self._components) > 1:
                # Compute mass fraction from mole (or vice versa)
                fracs = np.array(self._fractions)
                if self._mode == "mole":
                    mass_fracs = fracs * mws / np.sum(fracs * mws)
                    other = f"{mass_fracs[i]:.4f}"
                else:
                    moles = fracs / mws
                    mole_fracs = moles / moles.sum()
                    other = f"{mole_fracs[i]:.4f}"
            else:
                other = "1.0000"
            other_item = QTableWidgetItem(other)
            other_item.setFlags(other_item.flags() & ~Qt.ItemIsEditable)
            other_item.setForeground(QBrush(QColor("#666666")))
            self._table.setItem(i, 3, other_item)

            # Remove button
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(24, 24)
            remove_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
            remove_btn.clicked.connect(lambda checked, r=i: self._remove_row(r))
            self._table.setCellWidget(i, 4, remove_btn)

        self._table.blockSignals(False)
        self._count_label.setText(str(len(self._components)))

    def _remove_row(self, row: int):
        """Remove a specific row."""
        if 0 <= row < len(self._components):
            name = self._components[row].name
            del self._components[row]
            del self._fractions[row]
            self._refresh_table()
            self._update_preview()
            self._msg(f"Removed: {name}")

    # ══ Preview Update ══

    def _update_preview(self):
        """Update mixture property preview."""
        import numpy as np

        n = len(self._components)
        if n == 0:
            self._sum_label.setText("—")
            self._mw_label.setText("—")
            self._tc_label.setText("—")
            self._pc_label.setText("—")
            self._omega_label.setText("—")
            return

        fracs = np.array(self._fractions)
        total = fracs.sum()
        self._sum_label.setText(f"{total:.6f}")

        if total <= 0:
            return

        fracs_norm = fracs / total

        # MW: simple weighted average
        mws = np.array([c.molecular_weight for c in self._components])
        mw_mix = np.dot(fracs_norm, mws)
        self._mw_label.setText(f"{mw_mix:.2f}")

        # Pseudo-critical properties (Kay's rule)
        tcs = np.array([c.critical_temperature or 300.0 for c in self._components])
        pcs = np.array([c.critical_pressure or 1e6 for c in self._components])
        omegas = np.array([c.acentric_factor or 0.0 for c in self._components])

        tc_mix = np.dot(fracs_norm, tcs)
        pc_mix = np.dot(fracs_norm, pcs)
        omega_mix = np.dot(fracs_norm, omegas)

        self._tc_label.setText(f"{tc_mix:.1f}")
        self._pc_label.setText(f"{pc_mix / 1e6:.3f}")
        self._omega_label.setText(f"{omega_mix:.4f}")

    # ══ Public API ══

    def load_mixture(self, components: List["Substance"],
                     fractions: List[float]):
        """Load a predefined mixture.

        Args:
            components: List of Substance objects.
            fractions: Corresponding mole (or mass) fractions.
        """
        if len(components) != len(fractions):
            raise ValueError("components and fractions must have same length")

        self._components = list(components)
        self._fractions = list(fractions)
        self._refresh_table()
        self._update_preview()

    @property
    def components(self) -> List["Substance"]:
        """Get the component list."""
        return list(self._components)

    @property
    def mole_fractions(self) -> List[float]:
        """Get mole fractions (converts from mass if needed)."""
        if not self._components:
            return []

        fracs = np.array(self._fractions)
        total = fracs.sum()
        if total <= 0:
            return [0.0] * len(self._components)
        fracs = fracs / total

        if self._mode == "mass":
            mws = np.array([c.molecular_weight for c in self._components])
            moles = fracs / mws
            moles /= moles.sum()
            return moles.tolist()
        return fracs.tolist()

    @property
    def mass_fractions(self) -> List[float]:
        """Get mass fractions (converts from mole if needed)."""
        if not self._components:
            return []

        fracs = np.array(self._fractions)
        total = fracs.sum()
        if total <= 0:
            return [0.0] * len(self._components)
        fracs = fracs / total

        if self._mode == "mole":
            mws = np.array([c.molecular_weight for c in self._components])
            masses = fracs * mws
            masses /= masses.sum()
            return masses.tolist()
        return fracs.tolist()

    def to_eos_params(self) -> tuple:
        """Convert to EoS-compatible parameters.

        Returns:
            Tuple of (comp_params list, mole_fractions list) for pt_flash etc.
            Returns (None, None) if EoS not available.
        """
        if not _HAS_SUBSTANCE_DB or create_mixture_eos is None:
            return None, None

        eos, comp_params, mole_fracs = create_mixture_eos(
            'pr', substances=self._components,
            mole_fractions=self.mole_fractions
        )
        return comp_params, mole_fracs

    def _msg(self, text: str):
        """Append a message to the log area."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._msg_text.append(f"[{ts}] {text}")
        # Auto-scroll
        scrollbar = self._msg_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Test without DB
    if _HAS_SUBSTANCE_DB:
        db = SubstanceDatabase()
    else:
        db = None

    dialog = MixtureEditorDialog(substance_db=db)

    # Pre-populate with some test components if Substance available
    if Substance:
        test_subs = [
            Substance(id="CH4", name="Methane", molecular_weight=16.043,
                      critical_temperature=190.56, critical_pressure=4.599e6,
                      acentric_factor=0.008),
            Substance(id="C2H6", name="Ethane", molecular_weight=30.07,
                      critical_temperature=305.32, critical_pressure=4.872e6,
                      acentric_factor=0.099),
        ]
        dialog.load_mixture(test_subs, [0.8, 0.2])
        print("Pre-loaded 80/20 methane/ethane mixture")

    if dialog.exec_():
        print(f"Components: {[c.name for c in dialog.components]}")
        print(f"Mole fractions: {[f'{f:.4f}' for f in dialog.mole_fractions]}")
        print(f"Mass fractions: {[f'{f:.4f}' for f in dialog.mass_fractions]}")
    else:
        print("Cancelled.")

    sys.exit(0)
