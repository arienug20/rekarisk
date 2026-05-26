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
from .substance_selector import SubstanceSelector
from .audit_viewer import AuditViewer

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

        # Substance selector → slots
        self._substance_selector.substance_selected.connect(self._on_substance_selected)

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
        self._tab_widget.removeTab(index)
        if hasattr(widget, "deleteLater"):
            widget.deleteLater()

    # ══════════════════════════════════════════════════════════════════════
    # Slots
    # ══════════════════════════════════════════════════════════════════════

    def _on_add_scenario(self, scenario_type: str):
        """Handle 'Add Scenario' from project panel context menu."""
        type_labels = {
            "dispersion": "Dispersion",
            "fire": "Fire",
            "explosion": "Explosion",
            "source_term": "Source Term",
        }
        label = type_labels.get(scenario_type, scenario_type.title())
        from datetime import datetime as dt
        name = f"{label} {dt.now().strftime('%H:%M')}"

        # Add to project panel
        if scenario_type == "dispersion":
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
        """Handle substance selection from the selector."""
        name = substance.name if substance else "?"
        self.statusBar().showMessage(f"Selected: {name}", 5000)

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
            f"<p><b>License:</b> MIT</p>"
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
