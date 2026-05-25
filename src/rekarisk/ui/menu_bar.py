"""
Rekarisk UI — Menu Bar.

Provides the main application menu bar with File, Edit, View, Tools, and Help menus.
Follows standard desktop application conventions.

Actions emit signals that are handled by MainWindow.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenuBar, QMenu


class RekariskMenuBar(QMenuBar):
    """Main application menu bar."""

    # Signals emitted when menu actions are triggered
    new_project = pyqtSignal()
    open_project = pyqtSignal()
    save_project = pyqtSignal()
    save_project_as = pyqtSignal()
    close_project = pyqtSignal()
    exit_app = pyqtSignal()

    undo_triggered = pyqtSignal()
    redo_triggered = pyqtSignal()
    preferences = pyqtSignal()

    show_substance_db = pyqtSignal()
    show_batch_runner = pyqtSignal()
    show_sensitivity = pyqtSignal()
    show_monte_carlo = pyqtSignal()

    show_help = pyqtSignal()
    show_about = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_menus()

    def _setup_menus(self):
        self._create_file_menu()
        self._create_edit_menu()
        self._create_view_menu()
        self._create_tools_menu()
        self._create_help_menu()

    # ── File Menu ──

    def _create_file_menu(self):
        menu = self.addMenu("&File")

        act = QAction("&New Project...", self)
        act.setShortcut(QKeySequence.StandardKey.New)
        act.setStatusTip("Create a new consequence analysis project")
        act.triggered.connect(self.new_project.emit)
        menu.addAction(act)

        act = QAction("&Open Project...", self)
        act.setShortcut(QKeySequence.StandardKey.Open)
        act.setStatusTip("Open an existing project file (*.caproj)")
        act.triggered.connect(self.open_project.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("&Save", self)
        act.setShortcut(QKeySequence.StandardKey.Save)
        act.setStatusTip("Save the current project")
        act.triggered.connect(self.save_project.emit)
        menu.addAction(act)

        act = QAction("Save &As...", self)
        act.setShortcut(QKeySequence.StandardKey.SaveAs)
        act.setStatusTip("Save the project with a new name")
        act.triggered.connect(self.save_project_as.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("&Close Project", self)
        act.setShortcut(QKeySequence("Ctrl+W"))
        act.setStatusTip("Close the current project")
        act.triggered.connect(self.close_project.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("E&xit", self)
        act.setShortcut(QKeySequence.StandardKey.Quit)
        act.setStatusTip("Exit Rekarisk")
        act.triggered.connect(self.exit_app.emit)
        menu.addAction(act)

    # ── Edit Menu ──

    def _create_edit_menu(self):
        menu = self.addMenu("&Edit")

        act = QAction("&Undo", self)
        act.setShortcut(QKeySequence.StandardKey.Undo)
        act.triggered.connect(self.undo_triggered.emit)
        menu.addAction(act)

        act = QAction("&Redo", self)
        act.setShortcut(QKeySequence.StandardKey.Redo)
        act.triggered.connect(self.redo_triggered.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("&Preferences...", self)
        act.setShortcut(QKeySequence.StandardKey.Preferences)
        act.triggered.connect(self.preferences.emit)
        menu.addAction(act)

    # ── View Menu ──

    def _create_view_menu(self):
        menu = self.addMenu("&View")
        # Submenus populated by MainWindow
        self._view_menu = menu

    def add_view_action(self, text: str, shortcut: str = "",
                        checkable: bool = False) -> QAction:
        """Add an action to the View menu (used by MainWindow for panels/toolbars)."""
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if checkable:
            act.setCheckable(True)
            act.setChecked(True)
        self._view_menu.addAction(act)
        return act

    # ── Tools Menu ──

    def _create_tools_menu(self):
        menu = self.addMenu("&Tools")

        act = QAction("Substance &Database...", self)
        act.setShortcut(QKeySequence("Ctrl+D"))
        act.setStatusTip("View and manage the substance database")
        act.triggered.connect(self.show_substance_db.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("&Batch Runner...", self)
        act.setShortcut(QKeySequence("Ctrl+B"))
        act.setStatusTip("Run multiple cases in sequence")
        act.triggered.connect(self.show_batch_runner.emit)
        menu.addAction(act)

        act = QAction("&Sensitivity Analysis...", self)
        act.setStatusTip("Parameter sensitivity analysis")
        act.triggered.connect(self.show_sensitivity.emit)
        menu.addAction(act)

        act = QAction("&Monte Carlo Simulation...", self)
        act.setStatusTip("Uncertainty analysis via Monte Carlo")
        act.triggered.connect(self.show_monte_carlo.emit)
        menu.addAction(act)

    # ── Help Menu ──

    def _create_help_menu(self):
        menu = self.addMenu("&Help")

        act = QAction("&User Manual", self)
        act.setShortcut(QKeySequence.StandardKey.HelpContents)
        act.setStatusTip("Open the user manual")
        act.triggered.connect(self.show_help.emit)
        menu.addAction(act)

        menu.addSeparator()

        act = QAction("&About Rekarisk", self)
        act.setStatusTip("About this application")
        act.triggered.connect(self.show_about.emit)
        menu.addAction(act)
