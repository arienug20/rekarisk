"""
Rekarisk UI — QRA Input Panel.

Quantitative Risk Assessment input panel with tabbed interface:
  - Tab 1: Event Tree — visual tree builder, node editor
  - Tab 2: Frequency — component selector, leak size, frequency lookup
  - Tab 3: Population — grid editor (day/night, indoor/outdoor)
  - Tab 4: Risk Criteria — criterion selection (TNO/HSE/CSC)
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QFormLayout, QCheckBox, QTextEdit, QHeaderView,
    QMessageBox, QSplitter, QRadioButton, QButtonGroup, QFrame,
    QGridLayout, QSlider,
)

from ..models.qra.failure_frequency import (
    FailureFrequencyDB, FrequencyClass, ComponentType, LeakSize,
    get_default_db, lookup_frequency, classify_frequency,
)
from ..models.qra.event_tree import (
    EventTree, EventTreeNode, Scenario, ConsequenceType,
    create_generic_vessel_tree, create_generic_pipeline_tree,
)
from ..models.qra.ignition_prob import (
    default_ignition_data, immediate_ignition_probability,
    delayed_ignition_probability, explosion_probability,
    combined_ignition_probability,
    SubstanceCategory, LocationType, CongestionLevel, ConfinementLevel,
    IgnitionModel,
)
from ..models.qra.societal_risk import FN_CRITERIA, FNCriterion


# ── Frequency Tab ─────────────────────────────────────────────────────

class FrequencyTab(QWidget):
    """Frequency database lookup and equipment selection."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db = get_default_db()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Component selection group ────────────────────────────
        comp_group = QGroupBox("Equipment Component")
        comp_form = QFormLayout(comp_group)

        self._comp_combo = QComboBox()
        components = sorted(
            [ct.value for ct in self._db.list_components()]
        )
        self._comp_combo.addItems(components)
        comp_form.addRow("Component Type:", self._comp_combo)

        self._leak_combo = QComboBox()
        leak_sizes = ["small", "medium", "large", "full_bore"]
        self._leak_combo.addItems(leak_sizes)
        self._leak_combo.setCurrentText("small")
        comp_form.addRow("Leak Size:", self._leak_combo)

        self._quantity_spin = QSpinBox()
        self._quantity_spin.setRange(1, 10000)
        self._quantity_spin.setValue(1)
        self._quantity_spin.setSuffix(" units")
        comp_form.addRow("Quantity:", self._quantity_spin)

        self._length_spin = QDoubleSpinBox()
        self._length_spin.setRange(0.001, 10000.0)
        self._length_spin.setValue(1.0)
        self._length_spin.setDecimals(3)
        self._length_spin.setSuffix(" km")
        self._length_spin.setEnabled(False)
        comp_form.addRow("Pipe Length:", self._length_spin)

        # Enable length field only for pipe
        self._comp_combo.currentTextChanged.connect(self._on_component_changed)

        self._lookup_btn = QPushButton("🔍 Lookup Frequency")
        self._lookup_btn.clicked.connect(self._on_lookup)
        comp_form.addRow("", self._lookup_btn)

        layout.addWidget(comp_group)

        # ── Result display ───────────────────────────────────────
        result_group = QGroupBox("Frequency Result")
        result_layout = QFormLayout(result_group)

        self._freq_label = QLabel("—")
        self._freq_label.setFont(QFont("monospace", 12))
        self._freq_label.setStyleSheet("color: #1565C0; font-weight: bold;")
        result_layout.addRow("Frequency:", self._freq_label)

        self._class_label = QLabel("—")
        result_layout.addRow("Class:", self._class_label)

        self._source_label = QLabel("—")
        self._source_label.setWordWrap(True)
        result_layout.addRow("Source:", self._source_label)

        self._total_label = QLabel("—")
        result_layout.addRow("Total (quantity×freq):", self._total_label)

        layout.addWidget(result_group)

        # ── Modification factors ─────────────────────────────────
        mod_group = QGroupBox("Modification Factors (optional)")
        mod_layout = QVBoxLayout(mod_group)

        self._mod_list = QTableWidget(0, 2)
        self._mod_list.setHorizontalHeaderLabels(["Factor", "Value"])
        self._mod_list.horizontalHeader().setStretchLastSection(True)
        self._mod_list.setMaximumHeight(150)

        btn_row = QHBoxLayout()
        self._add_mod_btn = QPushButton("+ Add Factor")
        self._add_mod_btn.clicked.connect(self._on_add_modifier)
        self._clear_mod_btn = QPushButton("Clear")
        self._clear_mod_btn.clicked.connect(self._on_clear_modifiers)
        btn_row.addWidget(self._add_mod_btn)
        btn_row.addWidget(self._clear_mod_btn)
        btn_row.addStretch()

        mod_layout.addWidget(self._mod_list)
        mod_layout.addLayout(btn_row)

        layout.addWidget(mod_group)

        # ── Adjusted frequency ───────────────────────────────────
        adj_group = QGroupBox("Adjusted Frequency")
        adj_layout = QFormLayout(adj_group)

        self._adj_label = QLabel("—")
        self._adj_label.setFont(QFont("monospace", 12))
        self._adj_label.setStyleSheet("color: #E65100; font-weight: bold;")
        adj_layout.addRow("Adjusted:", self._adj_label)

        self._calc_adj_btn = QPushButton("📐 Calculate Adjusted")
        self._calc_adj_btn.clicked.connect(self._on_calculate_adjusted)
        adj_layout.addRow("", self._calc_adj_btn)

        layout.addWidget(adj_group)
        layout.addStretch()

    def _on_component_changed(self, text: str) -> None:
        """Enable pipe length field only for pipe components."""
        self._length_spin.setEnabled(text == "pipe")

    def _on_lookup(self) -> None:
        """Look up frequency from database."""
        comp = self._comp_combo.currentText()
        leak = self._leak_combo.currentText()
        qty = self._quantity_spin.value()
        length = self._length_spin.value()

        freq = lookup_frequency(comp, leak)
        freq_class = classify_frequency(freq)

        # For pipe, frequency scales with length
        if comp == "pipe":
            freq = freq * length

        total = freq * qty

        self._freq_label.setText(f"{freq:.4e} /yr")
        self._class_label.setText(f"{freq_class.value} ({freq_class.name})")
        self._total_label.setText(f"{total:.4e} /yr ({qty} × {freq:.4e})")

        entry = self._db.lookup_entry(comp, leak)
        if entry:
            source_text = f"{entry.source.value if hasattr(entry.source, 'value') else entry.source}"
            if entry.notes:
                source_text += f" — {entry.notes}"
            self._source_label.setText(source_text)
        else:
            self._source_label.setText("No entry found (default minimal value used)")

    def _on_add_modifier(self) -> None:
        """Add a row to the modifier table."""
        row = self._mod_list.rowCount()
        self._mod_list.insertRow(row)
        name_item = QTableWidgetItem("")
        value_item = QTableWidgetItem("1.0")
        self._mod_list.setItem(row, 0, name_item)
        self._mod_list.setItem(row, 1, value_item)

    def _on_clear_modifiers(self) -> None:
        """Clear all modifier rows."""
        self._mod_list.setRowCount(0)
        self._adj_label.setText("—")

    def _on_calculate_adjusted(self) -> None:
        """Calculate adjusted frequency with modifiers."""
        freq_text = self._freq_label.text().split()[0]
        try:
            base_freq = float(freq_text)
        except ValueError:
            self._adj_label.setText("Error: run lookup first")
            return

        adjusted = base_freq
        for row in range(self._mod_list.rowCount()):
            value_item = self._mod_list.item(row, 1)
            if value_item:
                try:
                    factor = float(value_item.text())
                    adjusted *= factor
                except ValueError:
                    pass

        self._adj_label.setText(f"{adjusted:.4e} /yr")

    def get_frequency(self) -> float:
        """Get the current frequency value."""
        text = self._adj_label.text()
        if text == "—":
            text = self._freq_label.text()
        try:
            return float(text.split()[0])
        except (ValueError, IndexError):
            return 0.0


# ── Event Tree Tab ────────────────────────────────────────────────────

class EventTreeTab(QWidget):
    """Event tree builder with visual tree display."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tree: Optional[EventTree] = None
        self._setup_ui()
        self._on_new_tree()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Tree controls ────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self._name_edit = QLineEdit("Pressurised Vessel Leak")
        self._name_edit.setPlaceholderText("Initiating event name...")
        ctrl_layout.addWidget(QLabel("Name:"))
        ctrl_layout.addWidget(self._name_edit)

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1e-12, 10.0)
        self._freq_spin.setValue(5e-6)
        self._freq_spin.setDecimals(10)
        self._freq_spin.setSingleStep(1e-6)
        self._freq_spin.setPrefix("f = ")
        self._freq_spin.setSuffix(" /yr")
        ctrl_layout.addWidget(self._freq_spin)

        new_btn = QPushButton("New Tree")
        new_btn.clicked.connect(self._on_new_tree)
        ctrl_layout.addWidget(new_btn)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # ── Quick templates ──────────────────────────────────────
        tmpl_layout = QHBoxLayout()
        tmpl_layout.addWidget(QLabel("Template:"))
        vessel_tmpl_btn = QPushButton("Vessel Leak")
        vessel_tmpl_btn.clicked.connect(self._on_vessel_template)
        tmpl_layout.addWidget(vessel_tmpl_btn)
        pipe_tmpl_btn = QPushButton("Pipeline Rupture")
        pipe_tmpl_btn.clicked.connect(self._on_pipe_template)
        tmpl_layout.addWidget(pipe_tmpl_btn)
        tmpl_layout.addStretch()

        calc_btn = QPushButton("⚡ Calculate Scenarios")
        calc_btn.clicked.connect(self._on_calculate)
        tmpl_layout.addWidget(calc_btn)
        layout.addLayout(tmpl_layout)

        # ── Tree display ─────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tree widget
        self._tree_widget = QTreeWidget()
        self._tree_widget.setHeaderLabels([
            "Node", "P(Yes)", "P(No)", "Outcome", "Type"
        ])
        self._tree_widget.setAlternatingRowColors(True)
        self._tree_widget.setMinimumHeight(200)

        # Scenario table
        self._scenario_table = QTableWidget(0, 5)
        self._scenario_table.setHorizontalHeaderLabels([
            "Scenario", "Frequency (/yr)", "Type", "Path", "Contribution %"
        ])
        self._scenario_table.horizontalHeader().setStretchLastSection(True)

        splitter.addWidget(self._tree_widget)
        splitter.addWidget(self._scenario_table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # ── Node editor ──────────────────────────────────────────
        editor_group = QGroupBox("Add Branch Node")
        editor_layout = QHBoxLayout(editor_group)

        editor_layout.addWidget(QLabel("Node Name:"))
        self._node_name_edit = QLineEdit()
        self._node_name_edit.setPlaceholderText("e.g., Immediate Ignition?")
        editor_layout.addWidget(self._node_name_edit)

        editor_layout.addWidget(QLabel("P(Yes):"))
        self._prob_yes_spin = QDoubleSpinBox()
        self._prob_yes_spin.setRange(0.0, 1.0)
        self._prob_yes_spin.setValue(0.1)
        self._prob_yes_spin.setSingleStep(0.05)
        self._prob_yes_spin.setDecimals(3)
        editor_layout.addWidget(self._prob_yes_spin)

        self._parent_combo = QComboBox()
        self._parent_combo.addItem("root")
        editor_layout.addWidget(QLabel("Parent:"))
        editor_layout.addWidget(self._parent_combo)

        add_btn = QPushButton("➕ Add Node")
        add_btn.clicked.connect(self._on_add_node)
        editor_layout.addWidget(add_btn)

        layout.addWidget(editor_group)

    def _on_new_tree(self) -> None:
        """Create a new empty event tree."""
        name = self._name_edit.text() or "New Event Tree"
        freq = self._freq_spin.value()
        self._tree = EventTree(name, freq)
        self._refresh_display()

    def _on_vessel_template(self) -> None:
        """Load generic vessel event tree template."""
        freq = self._freq_spin.value()
        name = self._name_edit.text() or "Pressurised Vessel Leak"
        self._tree = create_generic_vessel_tree(name, freq)
        self._refresh_display()

    def _on_pipe_template(self) -> None:
        """Load generic pipeline event tree template."""
        freq = self._freq_spin.value()
        name = self._name_edit.text() or "Pipeline Rupture"
        self._tree = create_generic_pipeline_tree(name, freq)
        self._refresh_display()

    def _on_add_node(self) -> None:
        """Add a branching node to the tree."""
        if not self._tree:
            QMessageBox.warning(self, "No Tree", "Create an event tree first.")
            return

        name = self._node_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Enter a node name.")
            return

        prob_yes = self._prob_yes_spin.value()
        parent = self._parent_combo.currentText()

        try:
            yes_node, no_node = self._tree.add_node(parent, name, prob_yes=prob_yes)
            # Update parent combo
            self._parent_combo.clear()
            names = []
            def _collect(node, depth):
                if node.name not in names:
                    # Use key-like naming
                    key = node.name.replace(": ", "_").replace(" ", "_").lower()
                    self._parent_combo.addItem(key)
                    names.append(key)
                for child in node.children:
                    _collect(child, depth + 1)
            _collect(self._tree.root, 0)
            self._refresh_display()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_calculate(self) -> None:
        """Calculate scenario probabilities."""
        if not self._tree:
            return
        self._refresh_scenarios()

    def _refresh_display(self) -> None:
        """Update the tree widget and scenario table."""
        if not self._tree:
            return
        self._refresh_tree_widget()
        self._refresh_scenarios()

    def _refresh_tree_widget(self) -> None:
        """Refresh the tree widget display."""
        self._tree_widget.clear()
        if not self._tree:
            return

        def _add_node(
            parent: QTreeWidgetItem,
            node: EventTreeNode,
            branch_label: str = "",
        ) -> None:
            name = f"{branch_label}{node.name}" if branch_label else node.name
            outcome = node.outcome_name if node.is_terminal else "—"
            ct = node.consequence_type.value if node.is_terminal else "—"
            prob_yes = f"{node.probability_yes:.4f}" if node != self._tree.root else "1.0000"
            prob_no = f"{node.prob_no:.4f}" if node != self._tree.root and not node.is_terminal else "—"

            item = QTreeWidgetItem([name, prob_yes, prob_no, outcome, ct])
            parent.addChild(item)

            for i, child in enumerate(node.children):
                prefix = "YES: " if i == 0 else "NO: "
                _add_node(item, child, prefix)

        root_item = QTreeWidgetItem([self._tree.name, "1.0000", "—", "—", "—"])
        self._tree_widget.addTopLevelItem(root_item)

        for child in self._tree.root.children:
            _add_node(root_item, child)

        self._tree_widget.expandAll()

    def _refresh_scenarios(self) -> None:
        """Refresh the scenarios table."""
        if not self._tree:
            return

        scenarios = self._tree.get_scenarios()
        self._scenario_table.setRowCount(len(scenarios))

        # Calculate total for percentages
        total_prob = sum(s.probability for s in scenarios)

        for i, scenario in enumerate(scenarios):
            self._scenario_table.setItem(i, 0, QTableWidgetItem(scenario.name))
            freq_item = QTableWidgetItem(f"{scenario.probability:.4e}")
            freq_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._scenario_table.setItem(i, 1, freq_item)
            self._scenario_table.setItem(i, 2, QTableWidgetItem(scenario.consequence_type.value))
            path_str = " → ".join(scenario.path) if scenario.path else "—"
            self._scenario_table.setItem(i, 3, QTableWidgetItem(path_str))

            pct = (scenario.probability / total_prob * 100) if total_prob > 0 else 0
            pct_item = QTableWidgetItem(f"{pct:.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._scenario_table.setItem(i, 4, pct_item)

        self._scenario_table.resizeColumnsToContents()

    def get_event_tree(self) -> Optional[EventTree]:
        """Get the current event tree."""
        return self._tree

    def get_scenarios(self) -> list:
        """Get calculated scenarios."""
        if self._tree:
            return self._tree.get_scenarios()
        return []


# ── Population Tab ────────────────────────────────────────────────────

class PopulationTab(QWidget):
    """Population grid editor with day/night and indoor/outdoor fractions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Grid configuration ───────────────────────────────────
        grid_config = QGroupBox("Grid Configuration")
        gc_layout = QHBoxLayout(grid_config)

        gc_layout.addWidget(QLabel("Grid Size:"))
        self._grid_rows = QSpinBox()
        self._grid_rows.setRange(2, 100)
        self._grid_rows.setValue(5)
        self._grid_rows.setPrefix("Rows: ")
        gc_layout.addWidget(self._grid_rows)

        self._grid_cols = QSpinBox()
        self._grid_cols.setRange(2, 100)
        self._grid_cols.setValue(5)
        self._grid_cols.setPrefix("Cols: ")
        gc_layout.addWidget(self._grid_cols)

        gc_layout.addWidget(QLabel("Cell Size:"))
        self._cell_size = QDoubleSpinBox()
        self._cell_size.setRange(1.0, 1000.0)
        self._cell_size.setValue(50.0)
        self._cell_size.setSuffix(" m")
        gc_layout.addWidget(self._cell_size)

        resize_btn = QPushButton("Resize Grid")
        resize_btn.clicked.connect(self._on_resize_grid)
        gc_layout.addWidget(resize_btn)

        gc_layout.addStretch()
        layout.addWidget(grid_config)

        # ── Population grid ──────────────────────────────────────
        pop_group = QGroupBox("Population (persons per cell)")
        pop_layout = QVBoxLayout(pop_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        self._day_radio = QRadioButton("Daytime")
        self._day_radio.setChecked(True)
        self._night_radio = QRadioButton("Nighttime")
        mode_layout.addWidget(self._day_radio)
        mode_layout.addWidget(self._night_radio)

        mode_layout.addWidget(QLabel("  Indoor fraction:"))
        self._indoor_spin = QDoubleSpinBox()
        self._indoor_spin.setRange(0.0, 1.0)
        self._indoor_spin.setValue(0.5)
        self._indoor_spin.setSingleStep(0.1)
        mode_layout.addWidget(self._indoor_spin)

        mode_layout.addWidget(QLabel("Outdoor:"))
        self._outdoor_label = QLabel("0.50")
        self._indoor_spin.valueChanged.connect(
            lambda v: self._outdoor_label.setText(f"{1-v:.2f}")
        )
        mode_layout.addWidget(self._outdoor_label)
        mode_layout.addStretch()

        pop_layout.addLayout(mode_layout)

        # Grid table
        self._pop_table = QTableWidget(5, 5)
        self._pop_table.setHorizontalHeaderLabels([f"Col {i}" for i in range(5)])
        self._pop_table.setVerticalHeaderLabels([f"Row {i}" for i in range(5)])
        self._on_resize_grid()  # Fill initial values

        pop_layout.addWidget(self._pop_table)

        # Quick fill
        fill_layout = QHBoxLayout()
        fill_layout.addWidget(QLabel("Fill all cells:"))
        self._fill_value = QDoubleSpinBox()
        self._fill_value.setRange(0.0, 10000.0)
        self._fill_value.setValue(1.0)
        self._fill_value.setSuffix(" persons")
        fill_layout.addWidget(self._fill_value)
        fill_btn = QPushButton("Fill")
        fill_btn.clicked.connect(self._on_fill_grid)
        fill_layout.addWidget(fill_btn)
        fill_layout.addStretch()
        pop_layout.addLayout(fill_layout)

        layout.addWidget(pop_group)

        # ── Total population display ─────────────────────────────
        total_group = QGroupBox("Summary")
        total_layout = QFormLayout(total_group)

        self._total_pop_label = QLabel("0.0")
        total_layout.addRow("Total Persons:", self._total_pop_label)

        self._total_day_label = QLabel("0.0")
        total_layout.addRow("Daytime:", self._total_day_label)

        self._total_night_label = QLabel("0.0")
        total_layout.addRow("Nighttime:", self._total_night_label)

        self._pop_table.cellChanged.connect(self._on_grid_changed)
        layout.addWidget(total_group)

    def _on_resize_grid(self) -> None:
        """Resize the population grid."""
        rows = self._grid_rows.value()
        cols = self._grid_cols.value()

        self._pop_table.blockSignals(True)
        self._pop_table.setRowCount(rows)
        self._pop_table.setColumnCount(cols)
        self._pop_table.setHorizontalHeaderLabels([f"Col {i}" for i in range(cols)])
        self._pop_table.setVerticalHeaderLabels([f"Row {i}" for i in range(rows)])

        # Fill with zeros
        for r in range(rows):
            for c in range(cols):
                if self._pop_table.item(r, c) is None:
                    self._pop_table.setItem(r, c, QTableWidgetItem("0.0"))
        self._pop_table.blockSignals(False)

        self._update_totals()

    def _on_fill_grid(self) -> None:
        """Fill all cells with the specified value."""
        value = self._fill_value.value()
        self._pop_table.blockSignals(True)
        for r in range(self._pop_table.rowCount()):
            for c in range(self._pop_table.columnCount()):
                self._pop_table.setItem(r, c, QTableWidgetItem(str(value)))
        self._pop_table.blockSignals(False)
        self._update_totals()

    def _on_grid_changed(self) -> None:
        self._update_totals()

    def _update_totals(self) -> None:
        """Update total population labels."""
        total = 0.0
        for r in range(self._pop_table.rowCount()):
            for c in range(self._pop_table.columnCount()):
                item = self._pop_table.item(r, c)
                if item:
                    try:
                        total += float(item.text())
                    except ValueError:
                        pass

        self._total_pop_label.setText(f"{total:.1f}")

        # Apply day/night indoor/outdoor fractions (placeholder)
        indoor_f = self._indoor_spin.value()
        self._total_day_label.setText(f"{total:.1f} ({indoor_f*100:.0f}% indoor)")
        self._total_night_label.setText(f"{total * 0.3:.1f} (30% of daytime)")

    def get_population_grid(self) -> list[list[float]]:
        """Return the population grid as a 2D list."""
        grid = []
        for r in range(self._pop_table.rowCount()):
            row = []
            for c in range(self._pop_table.columnCount()):
                item = self._pop_table.item(r, c)
                if item:
                    try:
                        row.append(float(item.text()))
                    except ValueError:
                        row.append(0.0)
                else:
                    row.append(0.0)
            grid.append(row)
        return grid

    def get_total_population(self) -> float:
        return sum(sum(row) for row in self.get_population_grid())


# ── Risk Criteria Tab ─────────────────────────────────────────────────

class RiskCriteriaTab(QWidget):
    """Risk criteria selection and configuration."""

    criteria_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._selected_criterion: Optional[FNCriterion] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── FN Criterion selection ───────────────────────────────
        fn_group = QGroupBox("FN Curve Criterion")
        fn_layout = QVBoxLayout(fn_group)

        self._criterion_buttons = QButtonGroup(self)
        for key, criterion in FN_CRITERIA.items():
            radio = QRadioButton(f"{criterion.name}\n({criterion.description[:80]}...)")
            radio.setToolTip(criterion.description)
            self._criterion_buttons.addButton(radio)
            fn_layout.addWidget(radio)
            if criterion == FN_CRITERIA.get("dutch_existing"):
                radio.setChecked(True)
                self._selected_criterion = criterion

        self._criterion_buttons.buttonClicked.connect(self._on_criterion_changed)
        layout.addWidget(fn_group)

        # ── IR thresholds ────────────────────────────────────────
        ir_group = QGroupBox("Individual Risk Thresholds")
        ir_layout = QFormLayout(ir_group)

        self._threshold_combo = QComboBox()
        self._threshold_combo.addItems(["HSE UK", "TNO Dutch", "NORSOK", "CCPS"])
        self._threshold_combo.currentTextChanged.connect(self.criteria_changed.emit)
        ir_layout.addRow("Standard:", self._threshold_combo)

        self._negligible_label = QLabel("1×10⁻⁶ /yr")
        ir_layout.addRow("Negligible:", self._negligible_label)

        self._acceptable_label = QLabel("1×10⁻⁵ /yr")
        ir_layout.addRow("Acceptable:", self._acceptable_label)

        self._tolerable_label = QLabel("1×10⁻⁴ /yr")
        ir_layout.addRow("Tolerable:", self._tolerable_label)

        self._intolerable_label = QLabel("1×10⁻³ /yr")
        ir_layout.addRow("Intolerable:", self._intolerable_label)

        # Update labels when standard changes
        self._threshold_combo.currentTextChanged.connect(self._on_threshold_standard_changed)

        layout.addWidget(ir_group)

        # ── Risk matrix type ─────────────────────────────────────
        matrix_group = QGroupBox("Risk Matrix")
        matrix_layout = QVBoxLayout(matrix_group)

        self._matrix_combo = QComboBox()
        self._matrix_combo.addItems(["ISO 17776", "API RP 752"])
        matrix_layout.addWidget(QLabel("Matrix Standard:"))
        matrix_layout.addWidget(self._matrix_combo)

        self._matrix_preview = QTextEdit()
        self._matrix_preview.setReadOnly(True)
        self._matrix_preview.setMaximumHeight(150)
        self._matrix_preview.setHtml(self._get_matrix_preview_html())
        matrix_layout.addWidget(self._matrix_preview)

        layout.addWidget(matrix_group)
        layout.addStretch()

    def _on_criterion_changed(self) -> None:
        """Update selected criterion."""
        btn = self._criterion_buttons.checkedButton()
        if btn:
            for key, criterion in FN_CRITERIA.items():
                if criterion.name in btn.text():
                    self._selected_criterion = criterion
                    break
        self.criteria_changed.emit()

    def _on_threshold_standard_changed(self, text: str) -> None:
        """Update IR threshold labels based on selected standard."""
        thresholds = {
            "HSE UK": ("1×10⁻⁶", "1×10⁻⁵", "1×10⁻⁴", "1×10⁻³"),
            "TNO Dutch": ("1×10⁻⁸", "1×10⁻⁶", "1×10⁻⁵", "1×10⁻⁵"),
            "NORSOK": ("1×10⁻⁶", "1×10⁻⁵", "1×10⁻⁴", "1×10⁻³"),
            "CCPS": ("1×10⁻⁶", "1×10⁻⁵", "1×10⁻⁴", "1×10⁻³"),
        }
        n, a, t, i = thresholds.get(text, thresholds["HSE UK"])
        self._negligible_label.setText(f"{n} /yr")
        self._acceptable_label.setText(f"{a} /yr")
        self._tolerable_label.setText(f"{t} /yr")
        self._intolerable_label.setText(f"{i} /yr")

    def _get_matrix_preview_html(self) -> str:
        """Generate HTML preview of the risk matrix."""
        from ..models.qra.risk_matrix import risk_matrix_html
        return risk_matrix_html(include_legend=False)

    def get_selected_criterion(self) -> Optional[FNCriterion]:
        return self._selected_criterion

    def get_ir_threshold_standard(self) -> str:
        return self._threshold_combo.currentText()


# ── Main QRA Panel ────────────────────────────────────────────────────

class QRAPanel(QWidget):
    """QRA input panel with tabbed interface."""

    calculate_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Tab widget
        self._tab_widget = QTabWidget()

        self._event_tree_tab = EventTreeTab()
        self._frequency_tab = FrequencyTab()
        self._population_tab = PopulationTab()
        self._criteria_tab = RiskCriteriaTab()

        self._tab_widget.addTab(self._event_tree_tab, "🌳 Event Tree")
        self._tab_widget.addTab(self._frequency_tab, "📊 Frequency")
        self._tab_widget.addTab(self._population_tab, "👥 Population")
        self._tab_widget.addTab(self._criteria_tab, "📏 Risk Criteria")

        layout.addWidget(self._tab_widget)

        # Run button
        run_layout = QHBoxLayout()
        run_layout.addStretch()
        self._run_btn = QPushButton("🚀 Run QRA Calculation")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; "
            "font-size: 14px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        run_layout.addWidget(self._run_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)

    def _on_run(self) -> None:
        """Emit calculate signal."""
        tree = self._event_tree_tab.get_event_tree()
        if not tree:
            QMessageBox.warning(self, "No Event Tree", "Build an event tree first.")
            return
        self.calculate_requested.emit()

    def set_substance(self, substance) -> None:
        """Pre-fill from a Substance database entry.

        QRA panel doesn't have direct substance fields; this is a no-op
        placeholder so the dispatch in main_window works without errors.
        """
        pass

    def get_event_tree(self) -> Optional[EventTree]:
        return self._event_tree_tab.get_event_tree()

    def get_scenarios(self) -> list:
        return self._event_tree_tab.get_scenarios()

    def get_frequency(self) -> float:
        return self._frequency_tab.get_frequency()

    def get_population_grid(self) -> list[list[float]]:
        return self._population_tab.get_population_grid()

    def get_total_population(self) -> float:
        return self._population_tab.get_total_population()

    def get_criterion(self) -> Optional[FNCriterion]:
        return self._criteria_tab.get_selected_criterion()

    def get_ir_standard(self) -> str:
        return self._criteria_tab.get_ir_threshold_standard()
