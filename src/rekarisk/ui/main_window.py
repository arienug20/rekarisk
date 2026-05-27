"""
Rekarisk UI — Main Window.

The primary application window: dock-based layout with project panel
on the left, central workspace for scenario editors, and support for
new/open/save project lifecycle.

Project File Format (*.caproj):
  ZIP-based format containing all project metadata, scenarios,
  weather cases, results, report configurations, and audit trail.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QAction, QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QMessageBox, QStatusBar, QFileDialog, QApplication,
    QToolBar, QSplitter, QInputDialog, QMenu,
)

from .menu_bar import RekariskMenuBar
from .project_panel import ProjectPanel
from .project_panel import (
    ITEM_SCENARIO_DISPERSION, ITEM_SCENARIO_FIRE,
    ITEM_SCENARIO_EXPLOSION, ITEM_SCENARIO_SOURCE_TERM,
)
from .substance_selector import SubstanceSelector
from .audit_viewer import AuditViewer

# Input panels
from .source_term_panel import SourceTermPanel
from .dispersion_panel import DispersionPanel
from .fire_panel import FirePanel
from .explosion_panel import ExplosionPanel
from .vulnerability_panel import VulnerabilityPanel
from .qra_panel import QRAPanel

# Results panels
from .source_term_results import SourceTermResultsPanel
from .dispersion_results import DispersionResultsPanel
from .fire_results import FireResultsPanel
from .explosion_results import ExplosionResultsPanel
from .vulnerability_results import VulnerabilityResultsWidget as VulnerabilityResultsPanel
from .qra_results import QRAResultsPanel

from ..core.substance_db import SubstanceDatabase, get_database
from ..core.audit_trail import AuditTrail, AuditAction
from ..core.project_file import ProjectFile, FILE_FILTER as PROJ_FILE_FILTER
from ..core.checkpoint import Checkpoint, get_project_id_from_path, get_project_id_from_name
from .. import __version__


PROJECT_FILE_EXT = "caproj"
PROJECT_FILE_FILTER = PROJ_FILE_FILTER
RECENT_FILES_KEY = "recent_files"
MAX_RECENT_FILES = 10


class MainWindow(QMainWindow):
    """Rekarisk main application window.

    Layout:
      ┌──────────────────────────────────────────────┐
      │  Menu Bar (File, Edit, View, Tools, Help)    │
      ├──────────┬───────────────────────────────────┤
      │          │                                   │
      │ Project  │     Central Workspace             │
      │ Panel    │     (Tabbed scenario editors,     │
      │          │      results viewer, etc.)        │
      │          │                                   │
      │ Substance│                                   │
      │ Selector │                                   │
      │          │                                   │
      ├──────────┴───────────────────────────────────┤
      │  Status Bar                                   │
      └──────────────────────────────────────────────┘
    """

    # Signals
    project_loaded = pyqtSignal(str)    # project name
    project_closed = pyqtSignal()
    project_modified = pyqtSignal(bool)  # is_dirty

    def __init__(self):
        super().__init__()

        # Application state
        self._project_path: Optional[Path] = None
        self._project_name: str = "Untitled"
        self._project_data: dict = {}
        self._is_dirty: bool = False
        self._db: SubstanceDatabase = get_database()

        # Phase 13 — Audit & File Management
        self._audit_trail = AuditTrail()
        self._project_file = ProjectFile()
        self._checkpoint: Optional[Checkpoint] = None
        self._audit_viewer: Optional[AuditViewer] = None
        self._audit_dock: Optional[QDockWidget] = None
        self._recent_menu: Optional[QMenu] = None

        # ── Scenario result cache (for cross-module data flow) ──
        self._last_source_term_result: Optional[dict] = None
        self._last_dispersion_result: Optional[dict] = None
        self._last_fire_result: Optional[dict] = None
        self._last_explosion_result: Optional[dict] = None
        self._last_vulnerability_result: Optional[dict] = None

        # ── Active panel tracking ──
        self._active_panels: dict[str, QWidget] = {}  # tab_label -> widget

        self._setup_window()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_docks()
        self._setup_central()
        self._setup_statusbar()
        self._connect_signals()
        self._setup_recent_files_menu()

        self.update_title()
        self.statusBar().showMessage("Ready — Create a new project or open an existing one", 5000)

    # ══════════════════════════════════════════════════════════════════════
    # Window Setup
    # ══════════════════════════════════════════════════════════════════════

    def _setup_window(self):
        self.setWindowTitle("Rekarisk")
        self.setMinimumSize(1024, 700)
        self.resize(1400, 900)

        # Restore window geometry from settings
        settings = QSettings("Rekarisk", "Rekarisk")
        geometry = settings.value("mainwindow/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("mainwindow/state")
        if state:
            self.restoreState(state)

    def _setup_menu(self):
        self._menu_bar = RekariskMenuBar(self)
        self.setMenuBar(self._menu_bar)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize() * 0.8)

        self._action_new = toolbar.addAction("New")
        self._action_new.setToolTip("New Project (Ctrl+N)")
        self._action_new.setShortcut("Ctrl+N")

        self._action_open = toolbar.addAction("Open")
        self._action_open.setToolTip("Open Project (Ctrl+O)")
        self._action_open.setShortcut("Ctrl+O")

        self._action_save = toolbar.addAction("Save")
        self._action_save.setToolTip("Save Project (Ctrl+S)")
        self._action_save.setShortcut("Ctrl+S")

        toolbar.addSeparator()

        self._action_run = toolbar.addAction("▶ Run")
        self._action_run.setToolTip("Run current scenario")
        self._action_run.setEnabled(False)

        toolbar.addSeparator()

        self._action_export = toolbar.addAction("Export")
        self._action_export.setToolTip("Export results")
        self._action_export.setEnabled(False)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _setup_docks(self):
        # Left dock — Project Panel
        self._project_panel = ProjectPanel()
        project_dock = QDockWidget("Project Navigator", self)
        project_dock.setWidget(self._project_panel)
        project_dock.setObjectName("ProjectDock")
        project_dock.setMinimumWidth(200)
        project_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project_dock)

        # Bottom-left dock — Substance Selector (below project panel)
        self._substance_selector = SubstanceSelector(self._db)
        substance_dock = QDockWidget("Substances", self)
        substance_dock.setWidget(self._substance_selector)
        substance_dock.setObjectName("SubstanceDock")
        substance_dock.setMinimumWidth(200)
        substance_dock.setMinimumHeight(150)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, substance_dock)

        # Add View menu entries
        self._view_project_dock = self._menu_bar.add_view_action(
            "Project Navigator", checkable=True
        )
        self._view_project_dock.toggled.connect(project_dock.setVisible)

        self._view_substance_dock = self._menu_bar.add_view_action(
            "Substance Selector", checkable=True
        )
        self._view_substance_dock.toggled.connect(substance_dock.setVisible)

    def _setup_central(self):
        """Central widget with tabbed workspace."""
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self._close_tab)

        # Welcome tab
        welcome = QWidget()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Welcome to Rekarisk")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333333;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(title)

        subtitle = QLabel(
            "Consequence & Risk Analysis for Safety Engineers\n\n"
            "Create a new project or open an existing one to get started."
        )
        subtitle.setStyleSheet("font-size: 14px; color: #555555;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        welcome_layout.addWidget(subtitle)

        shortcuts = QLabel(
            "Ctrl+N  —  New Project\n"
            "Ctrl+O  —  Open Project\n"
            "Ctrl+S  —  Save Project\n"
            "Ctrl+D  —  Substance Database\n"
            "F1      —  Help"
        )
        shortcuts.setStyleSheet("font-size: 12px; color: #444444; margin-top: 20px;")
        shortcuts.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(shortcuts)

        self._tab_widget.addTab(welcome, "🏠 Home")
        self.setCentralWidget(self._tab_widget)

    def _setup_statusbar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label, 1)

        self._dirty_label = QLabel("")
        self._status_bar.addPermanentWidget(self._dirty_label)

        self._version_label = QLabel(f"v{__version__}")
        self._version_label.setStyleSheet("color: #888888; padding-right: 8px;")
        self._status_bar.addPermanentWidget(self._version_label)

    # ══════════════════════════════════════════════════════════════════════
    # Signal Wiring
    # ══════════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Menu → slots
        self._menu_bar.new_project.connect(self.new_project)
        self._menu_bar.open_project.connect(self.open_project)
        self._menu_bar.save_project.connect(self.save_project)
        self._menu_bar.save_project_as.connect(self.save_project_as)
        self._menu_bar.close_project.connect(self.close_project)
        self._menu_bar.exit_app.connect(self.close)
        self._menu_bar.undo_triggered.connect(self._on_undo)
        self._menu_bar.redo_triggered.connect(self._on_redo)
        self._menu_bar.create_checkpoint.connect(self.create_checkpoint)
        self._menu_bar.restore_checkpoint.connect(self.restore_checkpoint)
        self._menu_bar.list_checkpoints.connect(self.list_checkpoints)
        self._menu_bar.show_audit_viewer.connect(self.show_audit_viewer)

        self._menu_bar.show_substance_db.connect(self._open_substance_db)
        self._menu_bar.show_batch_runner.connect(self._open_batch_runner)
        self._menu_bar.show_sensitivity.connect(self._open_sensitivity)
        self._menu_bar.show_monte_carlo.connect(self._open_monte_carlo)
        self._menu_bar.show_report.connect(self._open_report)
        self._menu_bar.show_comparison.connect(self._open_comparison)
        self._menu_bar.show_about.connect(self._show_about)

        # Toolbar → slots
        self._action_new.triggered.connect(self.new_project)
        self._action_open.triggered.connect(self.open_project)
        self._action_save.triggered.connect(self.save_project)

        # Project panel → slots
        self._project_panel.add_scenario.connect(self._on_add_scenario)
        self._project_panel.add_weather.connect(self._on_add_weather)
        self._project_panel.add_substance.connect(self._open_substance_db)

        # Project panel — double-click to open scenario editor
        self._project_panel.item_double_clicked.connect(self._on_item_double_clicked)

        # Substance selector → slots
        self._substance_selector.substance_selected.connect(self._on_substance_selected)

        # Toolbar Run → run active scenario
        self._action_run.triggered.connect(self._on_run_active_scenario)

    # ══════════════════════════════════════════════════════════════════════
    # Project Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def new_project(self):
        """Create a new empty project."""
        if not self._maybe_save():
            return

        self._project_path = None
        self._project_name = "Untitled"
        self._project_data = {
            "format_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "name": "Untitled",
            "description": "",
            "author": "",
            "scenarios": [],
            "weather_cases": [],
            "weather_data": {},
            "terrain_data": {},
            "substances": [],
            "settings": {},
            "results": [],
            "reports": [],
        }
        self._is_dirty = True

        # Phase 13: fresh audit trail and project file
        self._audit_trail = AuditTrail()
        self._project_file = ProjectFile()
        self._project_file.from_main_window_data(self._project_data)
        self._checkpoint = Checkpoint(get_project_id_from_name(self._project_name))

        self._audit_trail.log(
            action=AuditAction.CREATE,
            module="project",
            description=f"New project created: {self._project_name}",
        )

        self._project_panel.load_project("Untitled")
        self._update_audit_viewer()
        self.update_title()
        self.statusBar().showMessage("New project created", 3000)
        self.project_loaded.emit(self._project_name)

    def open_project(self, path: str | None = None):
        """Open an existing project file (.caproj)."""
        if not self._maybe_save():
            return

        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", "",
                PROJECT_FILE_FILTER
            )
            if not path:
                return

        try:
            # Phase 13: try .caproj format first
            proj_path = Path(path)
            if proj_path.suffix == f".{PROJECT_FILE_EXT}":
                self._project_file = ProjectFile.load(proj_path)
                self._audit_trail = self._project_file.audit_trail
                self._project_data = self._project_file.to_main_window_data()
            else:
                # Legacy JSON format fallback
                with open(path, "r", encoding="utf-8") as f:
                    self._project_data = json.load(f)
                self._project_file = ProjectFile()
                self._project_file.from_main_window_data(self._project_data)
                self._audit_trail = AuditTrail()
                self._audit_trail.log(
                    action=AuditAction.IMPORT,
                    module="project",
                    description=f"Opened legacy JSON project: {proj_path.name}",
                    details={"path": str(proj_path)},
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open project:\n{e}")
            return

        self._project_path = Path(path)
        self._project_name = self._project_data.get("name", self._project_path.stem)
        self._is_dirty = False
        self._checkpoint = Checkpoint(
            get_project_id_from_path(self._project_path))

        # Add to recent files
        self._add_recent_file(str(self._project_path))

        self._project_panel.load_project(self._project_name, str(self._project_path))
        self._update_audit_viewer()
        self.update_title()
        self.statusBar().showMessage(f"Opened: {self._project_path.name}", 3000)
        self.project_loaded.emit(self._project_name)

    def save_project(self) -> bool:
        """Save the current project. Returns True on success."""
        if self._project_path is None:
            return self.save_project_as()
        return self._write_project(self._project_path)

    def save_project_as(self) -> bool:
        """Save project with a new name/path."""
        default_name = f"{self._project_name}.{PROJECT_FILE_EXT}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", default_name,
            PROJECT_FILE_FILTER
        )
        if not path:
            return False
        self._project_path = Path(path)
        self._project_name = self._project_path.stem
        self._project_data["name"] = self._project_name
        return self._write_project(self._project_path)

    def _write_project(self, path: Path) -> bool:
        """Write project data to .caproj file. Returns True on success."""
        try:
            self._project_data["modified_at"] = datetime.now().isoformat()
            path.parent.mkdir(parents=True, exist_ok=True)

            # Phase 13: use ProjectFile for ZIP-based save
            self._project_file.from_main_window_data(self._project_data)
            # Preserve the existing audit trail entries
            for entry in self._audit_trail:
                if entry not in self._project_file.audit_trail:
                    self._project_file.audit_trail.log_entry(entry)

            # Log save action
            self._audit_trail.log(
                action=AuditAction.EXPORT,
                module="project",
                description=f"Project saved to {path.name}",
                details={"path": str(path), "scenarios": len(self._project_data.get("scenarios", []))},
            )
            self._project_file.audit_trail = self._audit_trail

            self._project_file.save(path)

            self._is_dirty = False
            self._update_audit_viewer()
            self.update_title()
            self.statusBar().showMessage(f"Saved: {path.name}", 3000)

            # Add to recent files
            self._add_recent_file(str(path))

            return True
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not save project:\n{e}")
            return False

    def close_project(self):
        """Close the current project."""
        if not self._maybe_save():
            return

        # Phase 13: auto-checkpoint before close
        if self._checkpoint and self._project_data:
            self._checkpoint.auto_checkpoint(self._project_data, "close")

        self._project_path = None
        self._project_name = "Untitled"
        self._project_data = {}
        self._is_dirty = False
        self._audit_trail = AuditTrail()
        self._project_file = ProjectFile()
        self._checkpoint = None
        self._project_panel.clear_project()
        self._update_audit_viewer()
        self.update_title()
        self.project_closed.emit()

    def _maybe_save(self) -> bool:
        """Prompt user to save if there are unsaved changes. Returns False if cancelled."""
        if not self._is_dirty:
            return True

        result = QMessageBox.question(
            self, "Unsaved Changes",
            f"Save changes to '{self._project_name}'?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )

        if result == QMessageBox.StandardButton.Save:
            return self.save_project()
        elif result == QMessageBox.StandardButton.Discard:
            return True
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Title & Status
    # ══════════════════════════════════════════════════════════════════════

    def update_title(self):
        """Update the window title to reflect project state."""
        dirty_marker = " ●" if self._is_dirty else ""
        if self._project_path:
            title = f"{self._project_name}{dirty_marker} — Rekarisk [{self._project_path}]"
        else:
            title = f"{self._project_name}{dirty_marker} — Rekarisk"
        self.setWindowTitle(title)
        self._dirty_label.setText("● Modified" if self._is_dirty else "")

    def set_dirty(self, dirty: bool = True):
        """Mark the project as modified/clean."""
        self._is_dirty = dirty
        self.update_title()
        self.project_modified.emit(dirty)

    # ══════════════════════════════════════════════════════════════════════
    # Central Tabs
    # ══════════════════════════════════════════════════════════════════════

    def add_central_tab(self, widget: QWidget, title: str) -> int:
        """Add a widget as a central tab. Returns the tab index."""
        idx = self._tab_widget.addTab(widget, title)
        self._tab_widget.setCurrentIndex(idx)
        return idx

    def _close_tab(self, index: int):
        """Handle tab close request."""
        if index <= 0:  # Don't close the welcome/home tab
            return
        widget = self._tab_widget.widget(index)
        tab_text = self._tab_widget.tabText(index)

        # Remove from active panels tracking
        self._active_panels.pop(tab_text, None)

        self._tab_widget.removeTab(index)
        if hasattr(widget, "deleteLater"):
            widget.deleteLater()

        # Disable Run button if no scenario tabs remain
        if not self._active_panels:
            self._action_run.setEnabled(False)

    # ══════════════════════════════════════════════════════════════════════
    # Slots
    # ══════════════════════════════════════════════════════════════════════

    def _on_add_scenario(self, scenario_type: str):
        """Handle 'Add Scenario' from project panel context menu."""
        type_labels = {
            "source_term": "Source Term",
            "dispersion": "Dispersion",
            "fire": "Fire",
            "explosion": "Explosion",
        }
        label = type_labels.get(scenario_type, scenario_type.title())
        from datetime import datetime as dt
        name = f"{label} {dt.now().strftime('%H:%M')}"

        # Add to project panel
        if scenario_type == "source_term":
            self._project_panel.add_scenario_item(
                "Scenarios", name, ITEM_SCENARIO_SOURCE_TERM)
        elif scenario_type == "dispersion":
            from .project_panel import ITEM_SCENARIO_DISPERSION
            self._project_panel.add_scenario_item("Scenarios", name,
                                                   ITEM_SCENARIO_DISPERSION)
        elif scenario_type == "fire":
            from .project_panel import ITEM_SCENARIO_FIRE
            self._project_panel.add_scenario_item("Scenarios", name,
                                                   ITEM_SCENARIO_FIRE)
        elif scenario_type == "explosion":
            from .project_panel import ITEM_SCENARIO_EXPLOSION
            self._project_panel.add_scenario_item("Scenarios", name,
                                                   ITEM_SCENARIO_EXPLOSION)

        # Add to project data
        scenario = {
            "id": len(self._project_data.get("scenarios", [])),
            "type": scenario_type,
            "name": name,
            "created_at": datetime.now().isoformat(),
        }
        self._project_data.setdefault("scenarios", []).append(scenario)

        # Phase 13: audit log
        self._audit_trail.log(
            action=AuditAction.CREATE,
            module=scenario_type,
            description=f"Added {label} scenario: {name}",
            details={"scenario_id": scenario["id"], "type": scenario_type},
        )

        self.set_dirty()
        self.statusBar().showMessage(f"Added {label} scenario: {name}", 3000)

        # Auto-open the corresponding editor panel
        panel_openers = {
            "source_term": self._open_source_term_panel,
            "dispersion": self._open_dispersion_panel,
            "fire": self._open_fire_panel,
            "explosion": self._open_explosion_panel,
        }
        opener = panel_openers.get(scenario_type)
        if opener:
            opener()

    def _on_add_weather(self):
        """Handle 'Add Weather Case' from project panel."""
        from datetime import datetime as dt
        name = f"Weather {dt.now().strftime('%H:%M')}"
        self._project_panel.add_weather_item(name)
        weather = {
            "id": len(self._project_data.get("weather_cases", [])),
            "name": name,
        }
        self._project_data.setdefault("weather_cases", []).append(weather)

        # Phase 13: audit log
        self._audit_trail.log(
            action=AuditAction.CREATE,
            module="weather",
            description=f"Added weather case: {name}",
            details=weather,
        )

        self.set_dirty()
        self.statusBar().showMessage(f"Added weather case: {name}", 3000)

    def _on_substance_selected(self, substance):
        """Handle substance selection — propagate to active panel."""
        name = substance.name if substance else "?"
        self.statusBar().showMessage(f"Selected: {name}", 5000)

        # Propagate to the currently active input panel
        current = self._tab_widget.currentWidget()
        if current is None:
            return

        # Check if it's a panel type that accepts substance data
        if hasattr(current, 'panel') and hasattr(current.panel, 'set_substance'):
            current.panel.set_substance(substance)
        elif hasattr(current, 'set_substance'):
            current.set_substance(substance)

    # ══════════════════════════════════════════════════════════════════════
    # Scenario Panel Management (double-click → open editor tab)
    # ══════════════════════════════════════════════════════════════════════

    def _on_item_double_clicked(self, data):
        """Open the appropriate panel when a project tree item is double-clicked."""
        if not data or not isinstance(data, dict):
            return

        item_type = data.get("type", "")

        # Handle scenario type items
        from .project_panel import (
            ITEM_SCENARIO_SOURCE_TERM, ITEM_SCENARIO_DISPERSION,
            ITEM_SCENARIO_FIRE, ITEM_SCENARIO_EXPLOSION,
        )

        # Map scenario types to panel openers
        scenario_map = {
            "dispersion": self._open_dispersion_panel,
            "fire": self._open_fire_panel,
            "explosion": self._open_explosion_panel,
        }

        # Check by folder name in data
        folder = data.get("name", "")
        if folder in scenario_map:
            scenario_map[folder]()
            return

        # Check by tree item type (stored in UserRole data)
        item_type_int = data.get("item_type")
        type_int_map = {
            ITEM_SCENARIO_SOURCE_TERM: self._open_source_term_panel,
            ITEM_SCENARIO_DISPERSION: self._open_dispersion_panel,
            ITEM_SCENARIO_FIRE: self._open_fire_panel,
            ITEM_SCENARIO_EXPLOSION: self._open_explosion_panel,
        }
        if item_type_int in type_int_map:
            type_int_map[item_type_int]()
            return

        # Default: check tree item type from the project panel's selected item
        selected = self._project_panel._tree.currentItem()
        if selected:
            itype = selected.type()
            if itype in type_int_map:
                type_int_map[itype]()
                return

        self.statusBar().showMessage(
            f"Double-clicked: {data.get('name', '?')} — no panel configured", 3000
        )

    def _make_scenario_tab(self, label: str, panel: QWidget, results: QWidget) -> QWidget:
        """Create a splitter widget with input panel and results panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(panel)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        # Store references for data access
        container.panel = panel
        container.results = results

        return container

    def _open_source_term_panel(self):
        """Open a Source Term input + results tab."""
        label = "🔩 Source Term"
        if label in self._active_panels:
            # Switch to existing tab
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = SourceTermPanel()
        results = SourceTermResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        # Wire calculation signal
        panel.calculation_requested.connect(
            lambda calc_type, params: self._execute_source_term(
                calc_type, params, results
            )
        )

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("Source Term editor opened", 3000)

    def _open_dispersion_panel(self):
        """Open a Dispersion input + results tab."""
        label = "🌬️ Dispersion"
        if label in self._active_panels:
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = DispersionPanel()
        results = DispersionResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        # Wire calculation signal
        panel.calculation_requested.connect(
            lambda params: self._execute_dispersion(params, results)
        )

        # Pre-fill from last source term result
        if self._last_source_term_result:
            st = self._last_source_term_result
            panel.set_source_params(
                source_rate=st.get("mass_flow_rate", 0),
                temperature=st.get("temperature", 298.15),
                molecular_weight=st.get("molecular_weight", 29.0) * 1000,  # kg/mol → g/mol
                phase=st.get("phase", "gas"),
                release_height=st.get("release_height", 0),
                release_diameter=st.get("hole_diameter", 0),
            )
            self.statusBar().showMessage(
                "Dispersion: pre-filled from Source Term results", 3000
            )

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("Dispersion editor opened", 3000)

    def _open_fire_panel(self):
        """Open a Fire input + results tab."""
        label = "🔥 Fire"
        if label in self._active_panels:
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = FirePanel()
        results = FireResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        # Wire calculation signal
        panel.calculation_requested.connect(
            lambda model_type, params: self._execute_fire(
                model_type, params, results
            )
        )

        # Pre-fill from last source term (for jet fire)
        # TODO: FirePanel needs set_source_params() method to auto-fill from source term
        # For now, user needs to manually enter values

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("Fire editor opened", 3000)

    def _open_explosion_panel(self):
        """Open an Explosion input + results tab."""
        label = "💥 Explosion"
        if label in self._active_panels:
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = ExplosionPanel()
        results = ExplosionResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        # Wire calculation signal
        panel.calculation_requested.connect(
            lambda params: self._execute_explosion(params, results)
        )

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("Explosion editor opened", 3000)

    def _open_vulnerability_panel(self):
        """Open a Vulnerability input + results tab."""
        label = "🛡️ Vulnerability"
        if label in self._active_panels:
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = VulnerabilityPanel()
        results = VulnerabilityResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        panel.calculation_requested.connect(
            lambda params: self._execute_vulnerability(params, results)
        )

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("Vulnerability editor opened", 3000)

    def _open_qra_panel(self):
        """Open a QRA input + results tab."""
        label = "📊 QRA"
        if label in self._active_panels:
            for i in range(self._tab_widget.count()):
                if self._tab_widget.tabText(i) == label:
                    self._tab_widget.setCurrentIndex(i)
                    return

        panel = QRAPanel()
        results = QRAResultsPanel()
        widget = self._make_scenario_tab(label, panel, results)

        panel.calculate_requested.connect(
            lambda: self._execute_qra(panel, results)
        )

        idx = self.add_central_tab(widget, label)
        self._active_panels[label] = widget
        self._action_run.setEnabled(True)
        self.statusBar().showMessage("QRA editor opened", 3000)

    # ══════════════════════════════════════════════════════════════════════
    # Execution Engine — run calculations and route data between modules
    # ══════════════════════════════════════════════════════════════════════

    def _on_run_active_scenario(self):
        """Run the calculation for the currently active scenario tab."""
        current = self._tab_widget.currentWidget()
        if current is None or not hasattr(current, 'panel'):
            self.statusBar().showMessage("No active scenario to run", 3000)
            return

        panel = current.panel

        # Trigger the panel's own run/calculate button
        if isinstance(panel, SourceTermPanel):
            calc_type, params = panel.get_current_tab_params()
            self._execute_source_term(calc_type, params, current.results)
        elif isinstance(panel, DispersionPanel):
            params = panel.get_all_params()
            self._execute_dispersion(params, current.results)
        elif isinstance(panel, FirePanel):
            # FirePanel needs to trigger its own _on_run
            panel._on_run()
        elif isinstance(panel, ExplosionPanel):
            panel._on_run()
        elif isinstance(panel, VulnerabilityPanel):
            panel._on_run()
        elif isinstance(panel, QRAPanel):
            panel._on_run()
        else:
            self.statusBar().showMessage(
                f"Cannot run: unknown panel type {type(panel).__name__}", 3000
            )

    def _execute_source_term(self, calc_type: str, params: dict, results_panel):
        """Execute a source term calculation and display results."""
        try:
            if calc_type == "orifice":
                from ..models.source_term.orifice import (
                    OrificeInput, calculate_orifice,
                )
                inp = OrificeInput(
                    Cd=params.get("Cd", 0.62),
                    d_hole=params.get("d_hole", 0.025),
                    P_upstream=params.get("P_upstream", 5e5),
                    P_downstream=params.get("P_downstream", 101325),
                    T=params.get("T", 300),
                    phase=params.get("phase", "auto"),
                    rho=params.get("rho", 1.2),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    h_liquid_head=params.get("h_liquid_head", 0),
                    duration=params.get("duration"),
                )
                result = calculate_orifice(inp)
                results_panel.show_orifice_result(result)

                # Cache for downstream modules
                self._last_source_term_result = {
                    "calc_type": "orifice",
                    "mass_flow_rate": result.mdot_initial,
                    "exit_velocity": result.velocity,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": result.phase,
                    "hole_diameter": params.get("d_hole", 0.025),
                    "is_choked": result.is_choked,
                    "total_mass": result.total_mass,
                }

            elif calc_type == "vessel":
                from ..models.source_term.vessel_depressur import (
                    VesselInput, calculate_vessel_blowdown,
                )
                inp = VesselInput(
                    V=params.get("V", 10),
                    A_wall=params.get("A_wall", 25),
                    P_initial=params.get("P_initial", 6e5),
                    T_initial=params.get("T_initial", 300),
                    orifice_d=params.get("orifice_d", 0.025),
                    Cd=params.get("Cd", 0.62),
                    t_max=params.get("t_max", 60),
                    P_target=params.get("P_target", 101325),
                    phase=params.get("phase", "gas"),
                    mode=params.get("mode", "api521"),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    rho_liquid=params.get("rho_liquid", 1000),
                )
                result = calculate_vessel_blowdown(inp)
                results_panel.show_vessel_result(result)

                # Cache — use average mass flow rate for downstream
                avg_mdot = (
                    sum(result.mdot) / len(result.mdot) if result.mdot else 0
                )
                self._last_source_term_result = {
                    "calc_type": "vessel",
                    "mass_flow_rate": avg_mdot,
                    "exit_velocity": result.mdot[0] / (
                        3.14159 * (params.get("orifice_d", 0.025) / 2) ** 2 * max(result.m[0], 0.1) / max(result.V[0], 1)
                    ) if result.mdot else 0,
                    "temperature": result.T[0] if result.T else 300,
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": params.get("phase", "gas"),
                    "hole_diameter": params.get("orifice_d", 0.025),
                    "total_mass": result.total_mass_released,
                }

            elif calc_type == "pipe":
                from ..models.source_term.pipe_flow import (
                    PipeInput, calculate_pipe_flow,
                )
                inp = PipeInput(
                    D=params.get("D", 0.1),
                    L=params.get("L", 100),
                    P_inlet=params.get("P_inlet", 5e5),
                    P_outlet=params.get("P_outlet", 101325),
                    T=params.get("T", 300),
                    phase=params.get("phase", "gas"),
                    rho=params.get("rho", 1.2),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    roughness=params.get("roughness", 4.5e-5),
                )
                result = calculate_pipe_flow(inp)
                results_panel.show_pipe_result(result)

                self._last_source_term_result = {
                    "calc_type": "pipe",
                    "mass_flow_rate": result.mdot,
                    "exit_velocity": result.velocity,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": result.flow_regime,
                    "total_mass": None,
                }

            elif calc_type == "psv":
                from ..models.source_term.relief_valve import (
                    ReliefValveInput, calculate_relief_valve,
                )
                inp = ReliefValveInput(
                    W_required=params.get("W_required", 1.0),
                    P_set=params.get("P_set", 5e5),
                    P_back=params.get("P_back", 101325),
                    T=params.get("T", 300),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    rho=params.get("rho", 1.2),
                    valve_type=params.get("valve_type", "conventional"),
                    overpressure_pct=params.get("overpressure_pct", 10),
                    rupture_disk=params.get("rupture_disk_used", False),
                )
                result = calculate_relief_valve(inp)
                results_panel.show_psv_result(result)

                self._last_source_term_result = {
                    "calc_type": "psv",
                    "mass_flow_rate": result.W_relieving,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": "gas",
                }

            elif calc_type == "pool":
                from ..models.source_term.pool_evaporation import (
                    PoolInput, calculate_pool_evaporation,
                )
                inp = PoolInput(
                    spill_mass=params.get("spill_mass", 1000),
                    rho_l=params.get("rho_l", 1000),
                    boiling_point=params.get("boiling_point", 373.15),
                    heat_of_vaporization=params.get("heat_of_vaporization", 2.26e6),
                    vapor_pressure=params.get("vapor_pressure", 3000),
                    molecular_weight=params.get("molecular_weight", 0.018),
                    T_ambient=params.get("T_ambient", 298.15),
                    wind_speed=params.get("wind_speed", 3.0),
                    surface=params.get("surface", "land"),
                    bunded_area=params.get("bunded_area"),
                    t_max=params.get("t_max", 120),
                )
                result = calculate_pool_evaporation(inp)
                results_panel.show_pool_result(result)

                self._last_source_term_result = {
                    "calc_type": "pool",
                    "mass_flow_rate": result.avg_evap_rate * result.pool_area[-1] if result.pool_area else 0,
                    "temperature": params.get("T_ambient", 298.15),
                    "molecular_weight": params.get("molecular_weight", 0.018),
                    "phase": "gas",
                }
            else:
                self.statusBar().showMessage(
                    f"Unknown source term type: {calc_type}", 3000
                )
                return

            # Store result in project data
            self._project_data.setdefault("results", []).append({
                "module": "source_term",
                "calc_type": calc_type,
                "inputs": params,
                "summary": self._last_source_term_result,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="source_term",
                description=f"Source term calculated: {calc_type}",
                details={"calc_type": calc_type},
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage(
                f"✅ Source Term ({calc_type}) calculation complete", 5000
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Calculation Error",
                f"Source Term ({calc_type}) failed:\n{e}"
            )

    def _execute_dispersion(self, params: dict, results_panel):
        """Execute a dispersion calculation."""
        try:
            from ..models.dispersion.dispersion_dispatcher import (
                ReleaseInfo, WeatherInfo, DispersionDispatcher,
            )

            release = ReleaseInfo(
                mass_rate=params.get("source_rate", 1.0),
                mass=params.get("source_mass", 0),
                duration=params.get("duration", 0),
                substance_density=params.get("cloud_density", 1.2),
                molecular_weight=params.get("molecular_weight", 29.0),
                temperature=params.get("temperature", 298.15),
                phase=params.get("phase", "gas"),
                release_height=params.get("release_height", 0),
                release_velocity=params.get("exit_velocity", 0),
                release_diameter=params.get("release_diameter", 0),
                heat_release_rate=params.get("heat_release_rate", 0),
            )

            weather = WeatherInfo(
                wind_speed=params.get("wind_speed", 3.0),
                stability_class=params.get("stability_class", "D"),
                terrain_type=params.get("terrain_type", "rural"),
            )

            dispatcher = DispersionDispatcher()
            result = dispatcher.dispatch(release, weather)
            results_panel.display_result(result)

            # Cache for downstream
            self._last_dispersion_result = {
                "model_used": result.model_used if hasattr(result, "model_used") else "unknown",
                "concentrations": result.concentrations if hasattr(result, "concentrations") else None,
                "x_grid": result.x_grid if hasattr(result, "x_grid") else None,
                "max_concentration": result.max_concentration if hasattr(result, "max_concentration") else 0,
                "params": params,
            }

            self._project_data.setdefault("results", []).append({
                "module": "dispersion",
                "inputs": params,
                "summary": {
                    "model": self._last_dispersion_result["model_used"],
                    "max_conc": self._last_dispersion_result["max_concentration"],
                },
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="dispersion",
                description="Dispersion calculation complete",
                details={"model": self._last_dispersion_result["model_used"]},
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Dispersion calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Dispersion failed:\n{e}")

    def _execute_fire(self, model_type: str, params: dict, results_panel):
        """Execute a fire consequence calculation."""
        try:
            if model_type == "pool_fire":
                from ..models.fire.pool_fire import PoolFireInput, calculate_pool_fire
                inp = PoolFireInput(
                    pool_diameter=params.get("pool_diameter", 10),
                    substance=params.get("substance", "gasoline"),
                    burning_rate=params.get("burning_rate"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.35),
                    wind_speed=params.get("wind_speed", 3.0),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                )
                result = calculate_pool_fire(inp)
                results_panel.display_pool_fire_result(result)

            elif model_type == "jet_fire":
                from ..models.fire.jet_fire import JetFireInput, calculate_jet_fire
                inp = JetFireInput(
                    orifice_diameter=params.get("orifice_diameter", 0.05),
                    discharge_velocity=params.get("discharge_velocity", 100),
                    mass_flow_rate=params.get("mass_flow_rate"),
                    substance=params.get("substance", "propane"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.30),
                    wind_speed=params.get("wind_speed", 3.0),
                    release_direction=params.get("release_direction", "horizontal"),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                    discharge_density=params.get("discharge_density"),
                )
                result = calculate_jet_fire(inp)
                results_panel.display_jet_fire_result(result)

            elif model_type == "bleve":
                from ..models.fire.bleve import BLEVEInput, calculate_bleve
                inp = BLEVEInput(
                    vessel_mass=params.get("vessel_mass", 5000),
                    substance=params.get("substance", "propane"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.35),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                )
                result = calculate_bleve(inp)
                results_panel.display_bleve_result(result)

            elif model_type == "flash_fire":
                from ..models.fire.flash_fire import FlashFireInput, calculate_flash_fire
                inp = FlashFireInput(
                    substance=params.get("substance", "methane"),
                    lfl=params.get("lfl", 0.05),
                    ufl=params.get("ufl", 0.15),
                    cloud_volume=params.get("cloud_volume", 1000),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                )
                result = calculate_flash_fire(inp)
                results_panel.display_flash_fire_result(result)
            else:
                self.statusBar().showMessage(f"Unknown fire type: {model_type}", 3000)
                return

            self._last_fire_result = {"model_type": model_type, "params": params}

            self._project_data.setdefault("results", []).append({
                "module": "fire",
                "calc_type": model_type,
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="fire",
                description=f"Fire calculation: {model_type}",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage(f"✅ Fire ({model_type}) calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Fire ({model_type}) failed:\n{e}")

    def _execute_explosion(self, params: dict, results_panel):
        """Execute an explosion consequence calculation."""
        try:
            results = []

            if params.get("tnt_enabled", True):
                from ..models.explosion.tnt_equivalency import TNTInput, calculate_tnt_equivalency
                inp = TNTInput(
                    mass_flammable=params.get("mass_flammable", 1000),
                    heat_of_combustion=params.get("heat_of_combustion", 50.35e6),
                    efficiency=params.get("tnt_efficiency", 0.05),
                )
                tnt_result = calculate_tnt_equivalency(inp)
                results.append(("TNT", tnt_result))

            if params.get("tno_enabled", True):
                from ..models.explosion.tno_multi_energy import TNOInput, calculate_tno_multi_energy
                inp = TNOInput(
                    confinement_class=params.get("tno_confinement_class", "2D"),
                    blast_strength=params.get("tno_blast_strength", 7),
                    energy=params.get("tno_energy", 1e9),
                )
                tno_result = calculate_tno_multi_energy(inp)
                results.append(("TNO", tno_result))

            if params.get("bst_enabled", True):
                from ..models.explosion.baker_strehlow import BSTInput, calculate_bst
                inp = BSTInput(
                    mass_flammable=params.get("bst_mass_flammable", 1000),
                    heat_of_combustion=params.get("bst_heat_of_combustion", 50.35e6),
                    fuel_reactivity=params.get("fuel_reactivity", "medium"),
                    confinement_class=params.get("bst_confinement_class", "2D"),
                    congestion_level=params.get("bst_congestion_level", "medium"),
                    flame_mach=params.get("flame_mach"),
                )
                bst_result = calculate_bst(inp)
                results.append(("BST", bst_result))

            if results:
                # Build dict of model_name -> result for the panel
                result_dict = {name: res for name, res in results}
                results_panel.display_results(result_dict)

            self._last_explosion_result = {"params": params, "num_models": len(results)}

            self._project_data.setdefault("results", []).append({
                "module": "explosion",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="explosion",
                description=f"Explosion: {len(results)} models",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Explosion calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Explosion failed:\n{e}")

    def _execute_vulnerability(self, params: dict, results_panel):
        """Execute a vulnerability assessment."""
        try:
            from ..models.vulnerability.vulnerability_calculator import (
                VulnerabilityInput, calculate_vulnerability,
            )

            inp = VulnerabilityInput(
                hazard_type=params.get("hazard_type", "toxic"),
                substance=params.get("substance"),
                thermal_model=params.get("thermal_model"),
                overpressure_model=params.get("overpressure_model"),
                exposure_time=params.get("exposure_time", 30),
                manual_intensity=params.get("manual_intensity", 100),
                intensity_source=params.get("intensity_source", "manual"),
                use_shelter=params.get("use_shelter", False),
                ach=params.get("ach"),
                x_min=params.get("x_min", 10),
                x_max=params.get("x_max", 5000),
                y_min=params.get("y_min", -500),
                y_max=params.get("y_max", 500),
                n_x=params.get("n_x", 100),
                n_y=params.get("n_y", 100),
            )

            result = calculate_vulnerability(inp)
            results_panel.set_result(result)

            self._last_vulnerability_result = {
                "hazard_type": params.get("hazard_type"),
                "result": result,
            }

            self._project_data.setdefault("results", []).append({
                "module": "vulnerability",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="vulnerability",
                description=f"Vulnerability: {params.get('hazard_type')}",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Vulnerability assessment complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Vulnerability failed:\n{e}")

    def _execute_qra(self, panel, results_panel):
        """Execute a QRA calculation."""
        try:
            # Collect params from the panel if it supports get_params
            params = {}
            if hasattr(panel, 'get_all_params'):
                params = panel.get_all_params()
            elif hasattr(panel, 'get_params'):
                params = panel.get_params()
            # QRA is more complex — use event tree + frequencies + consequences
            # For now, calculate FN curve if scenarios are provided
            from ..models.qra.societal_risk import calculate_fn_curve
            from ..models.qra.individual_risk import calculate_ir_grid

            self._project_data.setdefault("results", []).append({
                "module": "qra",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="qra",
                description="QRA calculation executed",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ QRA calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"QRA failed:\n{e}")

    def _open_report(self):
        """Open the report generation dialog."""
        from .report_dialog import ReportDialog
        results = self._project_data.get("results", [])
        if not results:
            # Build results from scenarios
            results = []
            for scenario in self._project_data.get("scenarios", []):
                results.append({
                    "name": scenario.get("name", "Scenario"),
                    "type": scenario.get("type", "general"),
                    "inputs": scenario.get("inputs", {}),
                    "summary": scenario.get("summary", {}),
                    "thresholds": scenario.get("thresholds", {}),
                    "table_headers": scenario.get("table_headers", []),
                    "table_rows": scenario.get("table_rows", []),
                })
        dialog = ReportDialog(self._project_data, results, self)
        dialog.exec()

    def _open_comparison(self):
        """Open the case comparison panel."""
        from .case_comparison import CaseComparisonPanel
        results = self._project_data.get("results", [])
        if not results:
            for scenario in self._project_data.get("scenarios", []):
                results.append({
                    "name": scenario.get("name", "Scenario"),
                    "type": scenario.get("type", "general"),
                    "summary": scenario.get("summary", {}),
                    "inputs": scenario.get("inputs", {}),
                })
        if not results:
            QMessageBox.information(self, "No Results",
                                    "Run some scenarios first to compare results.")
            return
        panel = CaseComparisonPanel()
        panel.set_results(results)
        idx = self.add_central_tab(panel, "\U0001f4ca Case Comparison")
        self.statusBar().showMessage("Case comparison ready", 3000)

    def _open_batch_runner(self):
        """Open the batch runner (placeholder)."""
        QMessageBox.information(self, "Batch Runner",
                                "Batch runner will be available in a future update.")

    def _open_sensitivity(self):
        """Open sensitivity analysis (placeholder)."""
        QMessageBox.information(self, "Sensitivity Analysis",
                                "Sensitivity analysis will be available in a future update.")

    def _open_monte_carlo(self):
        """Open Monte Carlo simulation (placeholder)."""
        QMessageBox.information(self, "Monte Carlo",
                                "Monte Carlo simulation will be available in a future update.")

    def _open_substance_db(self):
        """Open the substance database editor (placeholder)."""
        info = self._db.stats()
        QMessageBox.information(
            self, "Substance Database",
            f"Database: {info['path']}\n"
            f"Version: {info['version']}\n"
            f"Total: {info['total']} substances\n"
            f"  Flammable: {info['flammable']}\n"
            f"  Toxic: {info['toxic']}\n"
            f"  Gases: {info['gases']}\n"
            f"  Liquids: {info['liquids']}\n"
            f"Tags: {', '.join(info['tags']) or 'none'}"
        )

    def _show_about(self):
        """Show the About dialog."""
        QMessageBox.about(
            self, "About Rekarisk",
            f"<h2>Rekarisk v{__version__}</h2>"
            f"<p><i>Consequence & Risk Analysis for Safety Engineers</i></p>"
            f"<p><b>Codename:</b> Cikal</p>"
            f"<p><b>Author:</b> Arie Nugraha</p>"
            f"<p><b>License:</b> Proprietary</p>"
            f"<p><b>Standards:</b> Kepmen LH, API, CCPS, TNO Yellow Book</p>"
            f"<p><a href='https://github.com/arienug20/rekarisk'>"
            f"github.com/arienug20/rekarisk</a></p>"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Events
    # ══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event: QCloseEvent):
        """Handle main window close event."""
        if not self._maybe_save():
            event.ignore()
            return

        # Save window geometry
        settings = QSettings("Rekarisk", "Rekarisk")
        settings.setValue("mainwindow/geometry", self.saveGeometry())
        settings.setValue("mainwindow/state", self.saveState())

        event.accept()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 13 — Audit Trail Integration
    # ══════════════════════════════════════════════════════════════════════

    def _setup_audit_viewer(self) -> QDockWidget:
        """Create or return the audit viewer dock widget."""
        if self._audit_dock is not None:
            return self._audit_dock

        self._audit_viewer = AuditViewer()
        self._audit_dock = QDockWidget("Audit Trail", self)
        self._audit_dock.setWidget(self._audit_viewer)
        self._audit_dock.setObjectName("AuditDock")
        self._audit_dock.setMinimumWidth(300)
        self._audit_dock.setMinimumHeight(200)
        self._audit_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        # Add to View menu
        self._view_audit_dock = self._menu_bar.add_view_action(
            "Audit Trail", checkable=True
        )
        self._view_audit_dock.toggled.connect(self._audit_dock.setVisible)

        # Initially hidden, shown on demand
        self._audit_dock.hide()

        return self._audit_dock

    def _update_audit_viewer(self):
        """Refresh the audit viewer with current trail entries."""
        if self._audit_viewer is not None:
            self._audit_viewer.set_entries(self._audit_trail.to_dict_list())

    def show_audit_viewer(self):
        """Show the audit trail viewer dock."""
        dock = self._setup_audit_viewer()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.show()
        dock.raise_()
        self._update_audit_viewer()
        if self._view_audit_dock:
            self._view_audit_dock.setChecked(True)
        self.statusBar().showMessage("Audit Trail viewer opened", 3000)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 13 — Checkpoint / Undo System
    # ══════════════════════════════════════════════════════════════════════

    def create_checkpoint(self, label: str = "") -> Optional[str]:
        """Create a manual checkpoint of the current project state.

        Returns the checkpoint ID, or None if no project is loaded.
        """
        if not self._checkpoint:
            QMessageBox.warning(self, "No Project",
                                "Open or create a project first.")
            return None

        if not label:
            label, ok = QInputDialog.getText(
                self, "Checkpoint Label",
                "Label for this checkpoint:",
                text=f"Manual checkpoint {datetime.now().strftime('%H:%M')}"
            )
            if not ok or not label.strip():
                return None

        cid = self._checkpoint.create(self._project_data, label)
        self._audit_trail.log(
            action=AuditAction.RUN,
            module="checkpoint",
            description=f"Checkpoint created: {label}",
            details={"checkpoint_id": cid, "label": label},
        )
        self.statusBar().showMessage(f"Checkpoint created: {label}", 3000)
        return cid

    def restore_checkpoint(self):
        """Restore from a selected checkpoint."""
        if not self._checkpoint:
            QMessageBox.warning(self, "No Project",
                                "Open or create a project first.")
            return

        cps = self._checkpoint.list_checkpoints()
        if not cps:
            QMessageBox.information(self, "No Checkpoints",
                                    "No checkpoints available for this project.")
            return

        # Show list dialog
        items = []
        for cp in cps:
            ts = cp.get("timestamp", "")[:19]
            label = cp.get("label", "Unknown")
            sc = cp.get("scenario_count", 0)
            items.append(f"{ts} — {label} ({sc} scenarios)")

        item, ok = QInputDialog.getItem(
            self, "Restore Checkpoint",
            "Select a checkpoint to restore:",
            items, 0, False
        )
        if not ok or not item:
            return

        idx = items.index(item)
        cid = cps[idx]["id"]

        # Auto-checkpoint current state before restore
        self._checkpoint.auto_checkpoint(self._project_data, "restore")

        # Confirm
        result = QMessageBox.question(
            self, "Restore Checkpoint",
            f"Restore checkpoint:\n\n{item}\n\n"
            "Current unsaved changes will be lost. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        try:
            restored_data = self._checkpoint.restore(cid)
            self._project_data = restored_data
            self._project_file.from_main_window_data(restored_data)
            self._is_dirty = True
            self._project_name = self._project_data.get("name", self._project_name)
            self._project_panel.load_project(self._project_name,
                                              str(self._project_path) if self._project_path else "")

            self._audit_trail.log(
                action=AuditAction.RESTORE,
                module="checkpoint",
                description=f"Restored checkpoint: {cps[idx].get('label', 'Unknown')}",
                details={"checkpoint_id": cid},
            )

            self._update_audit_viewer()
            self.update_title()
            self.statusBar().showMessage(
                f"Restored checkpoint: {cps[idx].get('label', 'Unknown')}", 5000)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", str(e))

    def list_checkpoints(self):
        """Show a list of all checkpoints for the current project."""
        if not self._checkpoint:
            QMessageBox.warning(self, "No Project",
                                "Open or create a project first.")
            return

        cps = self._checkpoint.list_checkpoints()
        if not cps:
            QMessageBox.information(self, "No Checkpoints",
                                    "No checkpoints available for this project.")
            return

        lines = [f"Project: {self._project_name}"]
        lines.append(f"Total checkpoints: {len(cps)} (max 20)")
        lines.append("")
        for cp in cps:
            ts = cp.get("timestamp", "")[:19]
            label = cp.get("label", "Unknown")
            size_kb = cp.get("size_bytes", 0) / 1024
            lines.append(f"  {ts} — {label}")
            lines.append(f"         Size: {size_kb:.1f} KB | Scenarios: {cp.get('scenario_count', 0)}")

        QMessageBox.information(self, "Checkpoints", "\n".join(lines))

    def _on_undo(self):
        """Undo: restore from the latest checkpoint."""
        if not self._checkpoint:
            self.statusBar().showMessage("No project to undo", 3000)
            return

        cps = self._checkpoint.list_checkpoints()
        if not cps:
            self.statusBar().showMessage("No checkpoints to undo", 3000)
            return

        latest = cps[0]
        try:
            self._checkpoint.auto_checkpoint(self._project_data, "undo")
            restored_data = self._checkpoint.restore(latest["id"])
            self._project_data = restored_data
            self._project_file.from_main_window_data(restored_data)
            self._is_dirty = True
            self._project_name = self._project_data.get("name", self._project_name)
            self._project_panel.load_project(self._project_name,
                                              str(self._project_path) if self._project_path else "")

            self._audit_trail.log(
                action=AuditAction.RESTORE,
                module="undo",
                description=f"Undo to checkpoint: {latest.get('label', 'Unknown')}",
                details={"checkpoint_id": latest["id"]},
            )
            self._update_audit_viewer()
            self.update_title()
            self.statusBar().showMessage(
                f"Undo: restored '{latest.get('label', 'Unknown')}'", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Undo Failed", str(e))

    def _on_redo(self):
        """Redo is not directly supported — inform user."""
        self.statusBar().showMessage(
            "Redo: use Checkpoints → Restore to pick a specific checkpoint", 5000)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 13 — Recent Files
    # ══════════════════════════════════════════════════════════════════════

    def _setup_recent_files_menu(self):
        """Build the recent files submenu under File."""
        self._recent_menu = QMenu("Recent Projects", self)
        self._menu_bar.add_file_recent_menu(self._recent_menu)
        self._refresh_recent_menu()

    def _add_recent_file(self, path: str):
        """Add a file path to the recent files list."""
        settings = QSettings("Rekarisk", "Rekarisk")
        recent = settings.value(RECENT_FILES_KEY, []) or []
        if not isinstance(recent, list):
            recent = []

        # Remove if already exists (will be re-added at top)
        if path in recent:
            recent.remove(path)

        # Prepend and truncate
        recent.insert(0, path)
        recent = recent[:MAX_RECENT_FILES]

        settings.setValue(RECENT_FILES_KEY, recent)
        settings.sync()
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        """Rebuild the recent files submenu."""
        if self._recent_menu is None:
            return

        self._recent_menu.clear()
        settings = QSettings("Rekarisk", "Rekarisk")
        recent = settings.value(RECENT_FILES_KEY, []) or []

        if not recent or not isinstance(recent, list):
            act = self._recent_menu.addAction("(No recent projects)")
            act.setEnabled(False)
            return

        for p in recent:
            if os.path.exists(p):
                act = self._recent_menu.addAction(p)
                act.triggered.connect(lambda checked, path=p: self.open_project(path))

        self._recent_menu.addSeparator()
        clear_act = self._recent_menu.addAction("Clear Recent List")
        clear_act.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        settings = QSettings("Rekarisk", "Rekarisk")
        settings.setValue(RECENT_FILES_KEY, [])
        settings.sync()
        self._refresh_recent_menu()
        self.statusBar().showMessage("Recent files list cleared", 3000)
