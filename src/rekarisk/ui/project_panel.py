"""
Rekarisk UI — Project Panel.

Tree-view navigator showing the project structure:
  - Project root
    - Scenarios (dispersion, fire, explosion, etc.)
    - Weather Cases
    - Substances
    - Results
    - Reports

Supports right-click context menus, drag-and-drop reordering,
and double-click to open/edit.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QVBoxLayout, QWidget,
    QLabel, QHeaderView, QAbstractItemView,
)


# Tree item types
ITEM_PROJECT = 0
ITEM_SCENARIO_FOLDER = 1
ITEM_SCENARIO_DISPERSION = 2
ITEM_SCENARIO_FIRE = 3
ITEM_SCENARIO_EXPLOSION = 4
ITEM_SCENARIO_SOURCE_TERM = 5
ITEM_WEATHER_FOLDER = 6
ITEM_WEATHER = 7
ITEM_SUBSTANCE_FOLDER = 8
ITEM_SUBSTANCE = 9
ITEM_RESULT_FOLDER = 10
ITEM_RESULT = 11
ITEM_REPORT = 12


class ProjectPanel(QWidget):
    """Left sidebar panel showing the project hierarchy."""

    # Signals
    item_selected = pyqtSignal(object)       # the tree item or data
    item_double_clicked = pyqtSignal(object)
    add_scenario = pyqtSignal(str)           # scenario type
    add_weather = pyqtSignal()
    add_substance = pyqtSignal()
    delete_item = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._project_loaded = False

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Project")
        header.setStyleSheet(
            "QLabel { font-weight: bold; padding: 8px; "
            "background: #f5f5f5; border-bottom: 1px solid #cccccc; }"
        )
        layout.addWidget(header)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self._tree.currentItemChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

        layout.addWidget(self._tree)

    # ── Public API ──

    def load_project(self, project_name: str, project_path: str = ""):
        """Populate the tree with a new project structure.

        Args:
            project_name: Display name of the project.
            project_path: File path of the project file.
        """
        self._tree.clear()

        # Root item
        root = QTreeWidgetItem(self._tree, [project_name], ITEM_PROJECT)
        root.setData(0, Qt.ItemDataRole.UserRole, {"type": "project", "path": project_path})
        root.setExpanded(True)

        # Scenarios folder
        scenarios = QTreeWidgetItem(root, ["Scenarios"], ITEM_SCENARIO_FOLDER)
        scenarios.setData(0, Qt.ItemDataRole.UserRole, {"type": "folder", "name": "scenarios"})
        QTreeWidgetItem(scenarios, ["Dispersion"], ITEM_SCENARIO_DISPERSION)
        QTreeWidgetItem(scenarios, ["Fire"], ITEM_SCENARIO_FIRE)
        QTreeWidgetItem(scenarios, ["Explosion"], ITEM_SCENARIO_EXPLOSION)

        # Weather folder
        weather = QTreeWidgetItem(root, ["Weather Cases"], ITEM_WEATHER_FOLDER)
        weather.setData(0, Qt.ItemDataRole.UserRole, {"type": "folder", "name": "weather"})

        # Substances folder
        substances = QTreeWidgetItem(root, ["Substances"], ITEM_SUBSTANCE_FOLDER)
        substances.setData(0, Qt.ItemDataRole.UserRole,
                           {"type": "folder", "name": "substances"})

        # Results folder
        results = QTreeWidgetItem(root, ["Results"], ITEM_RESULT_FOLDER)
        results.setData(0, Qt.ItemDataRole.UserRole, {"type": "folder", "name": "results"})

        # Reports folder
        reports = QTreeWidgetItem(root, ["Reports"], ITEM_REPORT)
        reports.setData(0, Qt.ItemDataRole.UserRole, {"type": "folder", "name": "reports"})

        QTreeWidgetItem(reports, ["Summary Report"], ITEM_REPORT)
        QTreeWidgetItem(reports, ["Detailed Report"], ITEM_REPORT)

        self._tree.expandAll()
        self._project_loaded = True

    def clear_project(self):
        """Clear the tree (project closed)."""
        self._tree.clear()
        self._project_loaded = False

    def add_scenario_item(self, parent_name: str, name: str,
                          scenario_type: int, data: dict | None = None):
        """Add a user-created scenario under the appropriate folder.

        Args:
            parent_name: Folder name ("Scenarios", "Weather Cases", etc.).
            scenario_type: ITEM_* constant.
            data: Optional metadata dict.
        """
        items = self._tree.findItems(parent_name, Qt.MatchFlag.MatchExactly, 0)
        if not items:
            return
        item = QTreeWidgetItem(items[0], [name], scenario_type)
        item.setData(0, Qt.ItemDataRole.UserRole, data or {})
        items[0].setExpanded(True)
        self._tree.setCurrentItem(item)

    def add_weather_item(self, name: str, data: dict | None = None):
        """Add a weather case."""
        self.add_scenario_item("Weather Cases", name, ITEM_WEATHER, data)

    def add_result_item(self, name: str, data: dict | None = None):
        """Add a result entry."""
        self.add_scenario_item("Results", name, ITEM_RESULT, data)

    def get_selected_data(self) -> dict | None:
        """Get the UserRole data of the selected item."""
        item = self._tree.currentItem()
        if item:
            return item.data(0, Qt.ItemDataRole.UserRole)
        return None

    # ── Slots ──

    def _on_selection_changed(self, current, previous):
        if current:
            data = current.data(0, Qt.ItemDataRole.UserRole)
            self.item_selected.emit(data)

    def _on_double_click(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        self.item_double_clicked.emit(data)

    def _on_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item:
            return

        item_type = item.type()
        menu = QMenu(self)

        if item_type == ITEM_PROJECT:
            menu.addAction("Save Project", lambda: None)
            menu.addAction("Close Project", lambda: None)

        elif item_type == ITEM_SCENARIO_FOLDER:
            menu.addAction("Add Dispersion Scenario",
                           lambda: self.add_scenario.emit("dispersion"))
            menu.addAction("Add Fire Scenario",
                           lambda: self.add_scenario.emit("fire"))
            menu.addAction("Add Explosion Scenario",
                           lambda: self.add_scenario.emit("explosion"))

        elif item_type == ITEM_WEATHER_FOLDER:
            menu.addAction("Add Weather Case", self.add_weather.emit)

        elif item_type == ITEM_SUBSTANCE_FOLDER:
            menu.addAction("Add Substance", self.add_substance.emit)

        elif item_type == ITEM_RESULT_FOLDER:
            menu.addAction("Export Results...", lambda: None)

        elif item_type in (ITEM_SCENARIO_DISPERSION, ITEM_SCENARIO_FIRE,
                           ITEM_SCENARIO_EXPLOSION, ITEM_WEATHER, ITEM_RESULT,
                           ITEM_REPORT):
            menu.addAction("Delete", lambda: self.delete_item.emit(item))
            menu.addAction("Rename...", lambda: None)

        menu.exec(self._tree.viewport().mapToGlobal(pos))
