"""
Rekarisk UI — Audit Trail Viewer.

A dockable PyQt6 panel that displays the project's audit trail in a
sortable/filterable table. Supports:
  - Table view: timestamp | action | module | description
  - Filtering by action type, module, and date range
  - Detail view with before/after diff
  - Export filtered log to JSON
  - Full-text search within descriptions and details
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import (
    Qt, pyqtSignal, QDateTime, QSortFilterProxyModel, QModelIndex,
    QAbstractTableModel,
)
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QPushButton, QLabel, QComboBox, QLineEdit, QDateTimeEdit,
    QTextEdit, QSplitter, QMessageBox, QFileDialog, QMenu,
    QGroupBox, QFormLayout, QCheckBox, QAbstractItemView,
)


# ── Color Coding for Action Types ──

ACTION_COLORS = {
    "create": QColor(76, 175, 80),      # green
    "modify": QColor(33, 150, 243),     # blue
    "delete": QColor(244, 67, 54),      # red
    "run": QColor(156, 39, 176),        # purple
    "export": QColor(255, 152, 0),      # orange
    "import": QColor(0, 150, 136),      # teal
    "clone": QColor(96, 125, 139),      # blue-grey
    "restore": QColor(121, 85, 72),     # brown
}


# ── Table Model ──

class AuditTableModel(QAbstractTableModel):
    """Model backing the audit trail table view."""

    COLUMNS = ["Timestamp", "Action", "Module", "Description"]
    COL_KEYS = ["timestamp", "action", "module", "description"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[Dict[str, Any]] = []
        self._all_entries: List[Dict[str, Any]] = []

    # ── Data ──

    def set_entries(self, entries: List[Dict[str, Any]]):
        """Replace all entries."""
        self.beginResetModel()
        self._entries = list(entries)
        self._all_entries = list(entries)
        self.endResetModel()

    def add_entry(self, entry: Dict[str, Any]):
        """Append a single entry."""
        self.beginInsertRows(QModelIndex(), len(self._entries), len(self._entries))
        self._entries.append(entry)
        self._all_entries.append(entry)
        self.endInsertRows()

    def clear(self):
        """Remove all entries."""
        self.beginResetModel()
        self._entries.clear()
        self._all_entries.clear()
        self.endResetModel()

    # ── Qt Model Interface ──

    def rowCount(self, parent=QModelIndex()):
        return len(self._entries)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        entry = self._entries[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            key = self.COL_KEYS[col]
            value = entry.get(key, "")

            if key == "timestamp":
                try:
                    dt = datetime.fromisoformat(str(value))
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    return str(value)
            return str(value)

        elif role == Qt.ItemDataRole.ForegroundRole:
            action = entry.get("action", "")
            if col == 1:  # action column
                return ACTION_COLORS.get(action, QColor(0, 0, 0))
            return None

        elif role == Qt.ItemDataRole.FontRole:
            if col == 1:  # bold for action
                font = QFont()
                font.setBold(True)
                return font
            return None

        elif role == Qt.ItemDataRole.UserRole:
            return entry

        elif role == Qt.ItemDataRole.ToolTipRole:
            details = entry.get("details", {})
            if details:
                return f"Details: {len(details)} fields"
            return None

        return None

    # ── Filtering ──

    def filter_by_action(self, action: str | None):
        """Filter entries by action type."""
        self.beginResetModel()
        if action is None or action == "All":
            self._entries = list(self._all_entries)
        else:
            self._entries = [e for e in self._all_entries
                             if e.get("action") == action]
        self.endResetModel()

    def filter_by_module(self, module: str | None):
        """Filter entries by module name."""
        self.beginResetModel()
        if module is None or module == "All":
            # Re-apply only module filter; keep other filters
            self._entries = list(self._all_entries)
        else:
            self._entries = [e for e in self._all_entries
                             if e.get("module") == module]
        self.endResetModel()

    def filter_by_date(self, since: Optional[datetime],
                       until: Optional[datetime]):
        """Filter entries within a date range."""
        self.beginResetModel()
        filtered = list(self._all_entries)
        if since is not None:
            filtered = [e for e in filtered
                        if _parse_ts(e.get("timestamp", "")) >= since]
        if until is not None:
            filtered = [e for e in filtered
                        if _parse_ts(e.get("timestamp", "")) <= until]
        self._entries = filtered
        self.endResetModel()

    def filter_by_text(self, text: str):
        """Full-text search in description and details."""
        self.beginResetModel()
        if not text.strip():
            self._entries = list(self._all_entries)
        else:
            t = text.lower()
            self._entries = [
                e for e in self._all_entries
                if t in str(e.get("description", "")).lower() or
                t in str(e.get("details", "")).lower()
            ]
        self.endResetModel()

    def reset_filters(self):
        """Remove all filters."""
        self.beginResetModel()
        self._entries = list(self._all_entries)
        self.endResetModel()

    def get_entry(self, row: int) -> Optional[Dict[str, Any]]:
        """Get the full entry dict at a given row."""
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def get_all_actions(self) -> List[str]:
        """Return sorted unique action types."""
        actions = {e.get("action", "") for e in self._all_entries if e.get("action")}
        return sorted(actions)

    def get_all_modules(self) -> List[str]:
        """Return sorted unique module names."""
        modules = {e.get("module", "") for e in self._all_entries if e.get("module")}
        return sorted(modules)


# ── Helper ──

def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO timestamp string safely."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


# ── Audit Viewer Panel ──

class AuditViewer(QWidget):
    """Dockable panel for viewing, filtering, and exporting the audit trail.

    Layout:
      ┌──────────────────────────────────────────┐
      │  Filter Bar                              │
      │  [Action ▼] [Module ▼] [Search...]      │
      │  [From: ...] [To: ...] [Reset] [Export] │
      ├──────────────────────────────────────────┤
      │  Table                                   │
      │  Timestamp │ Action │ Module │ Desc      │
      ├──────────────────────────────────────────┤
      │  Detail Pane                             │
      │  Shows before/after diff, full details   │
      └──────────────────────────────────────────┘
    """

    # Signals
    export_requested = pyqtSignal(str)   # path to export to

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Filter Bar ──
        filter_bar = QHBoxLayout()

        # Action filter
        filter_bar.addWidget(QLabel("Action:"))
        self._action_filter = QComboBox()
        self._action_filter.addItem("All")
        self._action_filter.setMinimumWidth(80)
        self._action_filter.currentTextChanged.connect(self._on_filter_changed)
        filter_bar.addWidget(self._action_filter)

        # Module filter
        filter_bar.addWidget(QLabel("Module:"))
        self._module_filter = QComboBox()
        self._module_filter.addItem("All")
        self._module_filter.setMinimumWidth(80)
        self._module_filter.currentTextChanged.connect(self._on_filter_changed)
        filter_bar.addWidget(self._module_filter)

        # Search
        filter_bar.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Type to search...")
        self._search_input.setMinimumWidth(120)
        self._search_input.textChanged.connect(self._on_search_changed)
        filter_bar.addWidget(self._search_input)

        filter_bar.addStretch()

        layout.addLayout(filter_bar)

        # ── Date Range ──
        date_bar = QHBoxLayout()

        date_bar.addWidget(QLabel("From:"))
        self._date_from = QDateTimeEdit()
        self._date_from.setCalendarPopup(True)
        self._date_from.setDisplayFormat("yyyy-MM-dd HH:mm")
        self._date_from.setSpecialValueText("Any")
        self._date_from.dateTimeChanged.connect(self._on_date_changed)
        date_bar.addWidget(self._date_from)

        date_bar.addWidget(QLabel("To:"))
        self._date_to = QDateTimeEdit()
        self._date_to.setCalendarPopup(True)
        self._date_to.setDisplayFormat("yyyy-MM-dd HH:mm")
        self._date_to.setSpecialValueText("Any")
        self._date_to.setDateTime(QDateTime.currentDateTime())
        self._date_to.dateTimeChanged.connect(self._on_date_changed)
        date_bar.addWidget(self._date_to)

        # Reset button
        self._reset_btn = QPushButton("Reset Filters")
        self._reset_btn.clicked.connect(self._on_reset_filters)
        date_bar.addWidget(self._reset_btn)

        # Export button
        self._export_btn = QPushButton("Export Log")
        self._export_btn.clicked.connect(self._on_export)
        date_bar.addWidget(self._export_btn)

        date_bar.addStretch()

        layout.addLayout(date_bar)

        # ── Splitter (Table + Detail) ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Table
        self._table = QTableView()
        self._model = AuditTableModel()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed)

        # Column sizing
        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        splitter.addWidget(self._table)

        # Detail pane
        detail_group = QGroupBox("Entry Details")
        detail_layout = QVBoxLayout(detail_group)
        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setMaximumHeight(200)
        self._detail_text.setStyleSheet(
            "QTextEdit { font-family: 'Courier New', monospace; font-size: 11px; }")
        detail_layout.addWidget(self._detail_text)
        splitter.addWidget(detail_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # ── Status ──
        self._status_label = QLabel("No entries")
        layout.addWidget(self._status_label)

    # ── Public API ──

    def set_entries(self, entries: List[Dict[str, Any]]):
        """Populate the viewer with audit entries (as dicts).

        Args:
            entries: List of entry dicts from AuditEntry.to_dict().
        """
        self._model.set_entries(entries)
        self._update_filter_combos()
        self._update_status()
        self._detail_text.clear()

    def add_entry(self, entry: Dict[str, Any]):
        """Append a single entry to the view."""
        self._model.add_entry(entry)
        self._update_filter_combos()
        self._update_status()

    def clear(self):
        """Clear all entries from the viewer."""
        self._model.clear()
        self._detail_text.clear()
        self._update_status()

    def get_filtered_entries(self) -> List[Dict[str, Any]]:
        """Return currently visible (filtered) entries."""
        entries = []
        for row in range(self._model.rowCount()):
            entry = self._model.get_entry(row)
            if entry:
                entries.append(entry)
        return entries

    # ── Slots ──

    def _on_filter_changed(self):
        """Apply action + module combined filter."""
        action = self._action_filter.currentText()
        module = self._module_filter.currentText()

        # Start with all entries
        all_entries = self._model._all_entries

        filtered = list(all_entries)
        if action != "All":
            filtered = [e for e in filtered if e.get("action") == action]
        if module != "All":
            filtered = [e for e in filtered if e.get("module") == module]

        # Also apply search if present
        search_text = self._search_input.text().strip()
        if search_text:
            t = search_text.lower()
            filtered = [
                e for e in filtered
                if t in str(e.get("description", "")).lower() or
                t in str(e.get("details", "")).lower()
            ]

        # Also apply date range
        since = self._get_date_from()
        until = self._get_date_to()
        if since is not None:
            filtered = [e for e in filtered
                        if _parse_ts(e.get("timestamp", "")) >= since]
        if until is not None:
            filtered = [e for e in filtered
                        if _parse_ts(e.get("timestamp", "")) <= until]

        self._model.beginResetModel()
        self._model._entries = filtered
        self._model.endResetModel()
        self._update_status()
        self._detail_text.clear()

    def _on_search_changed(self, text: str):
        """Search by text in description/details (combined with other filters)."""
        self._on_filter_changed()

    def _on_date_changed(self):
        """Re-filter when date range changes."""
        self._on_filter_changed()

    def _on_reset_filters(self):
        """Reset all filters to defaults."""
        self._action_filter.setCurrentIndex(0)
        self._module_filter.setCurrentIndex(0)
        self._search_input.clear()
        self._date_from.clear()
        self._date_to.setDateTime(QDateTime.currentDateTime())
        self._model.reset_filters()
        self._update_status()
        self._detail_text.clear()

    def _on_selection_changed(self):
        """Show details of the selected entry."""
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            self._detail_text.clear()
            return

        entry = self._model.get_entry(indexes[0].row())
        if not entry:
            return

        lines = []

        # Header
        lines.append(f"Entry ID:   {entry.get('entry_id', 'N/A')}")
        lines.append(f"Timestamp:  {entry.get('timestamp', 'N/A')}")
        lines.append(f"Action:     {entry.get('action', 'N/A')}")
        lines.append(f"Module:     {entry.get('module', 'N/A')}")
        lines.append(f"User:       {entry.get('user', 'N/A')}")
        lines.append(f"Checksum:   {entry.get('checksum', 'N/A')[:16]}...")
        lines.append(f"Description: {entry.get('description', 'N/A')}")
        lines.append("")

        # Details
        details = entry.get("details", {})
        if details:
            lines.append(f"--- Details ({len(details)} fields) ---")

            # Highlight before/after
            if "before" in details:
                lines.append("")
                lines.append("  [BEFORE]")
                lines.append(_format_value(details["before"], indent=4))

            if "after" in details:
                lines.append("")
                lines.append("  [AFTER]")
                lines.append(_format_value(details["after"], indent=4))

            # Other details
            other = {k: v for k, v in details.items()
                     if k not in ("before", "after")}
            if other:
                lines.append("")
                lines.append("  [OTHER]")
                for k, v in other.items():
                    lines.append(f"    {k}: {_format_value(v, indent=6)}")

        else:
            lines.append("(no details)")

        self._detail_text.setPlainText("\n".join(lines))

    def _on_export(self):
        """Export filtered log to JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Log", "audit_log.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        import json
        entries = self.get_filtered_entries()
        data = {
            "format": "rekarisk-audit-1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": entries,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.export_requested.emit(path)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(entries)} entries to:\n{path}")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_context_menu(self, pos):
        """Right-click context menu on the table."""
        menu = QMenu(self)

        copy_action = menu.addAction("Copy Entry Details")
        copy_all_action = menu.addAction("Copy All (Filtered) Entries")

        action = menu.exec(self._table.viewport().mapToGlobal(pos))

        if action == copy_action:
            self._detail_text.selectAll()
            self._detail_text.copy()

        elif action == copy_all_action:
            import json
            entries = self.get_filtered_entries()
            text = json.dumps(entries, indent=2, ensure_ascii=False, default=str)
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)

    # ── Helpers ──

    def _update_filter_combos(self):
        """Update the action and module filter dropdowns."""
        current_action = self._action_filter.currentText()
        current_module = self._module_filter.currentText()

        self._action_filter.blockSignals(True)
        self._module_filter.blockSignals(True)

        # Update actions
        self._action_filter.clear()
        self._action_filter.addItem("All")
        for action in self._model.get_all_actions():
            self._action_filter.addItem(action)

        # Update modules
        self._module_filter.clear()
        self._module_filter.addItem("All")
        for module in self._model.get_all_modules():
            self._module_filter.addItem(module)

        # Restore previous selection if still valid
        idx = self._action_filter.findText(current_action)
        if idx >= 0:
            self._action_filter.setCurrentIndex(idx)

        idx = self._module_filter.findText(current_module)
        if idx >= 0:
            self._module_filter.setCurrentIndex(idx)

        self._action_filter.blockSignals(False)
        self._module_filter.blockSignals(False)

    def _update_status(self):
        """Update the status bar label."""
        total = len(self._model._all_entries)
        visible = self._model.rowCount()
        if total == 0:
            self._status_label.setText("No entries")
        elif total == visible:
            self._status_label.setText(f"{total} entries")
        else:
            self._status_label.setText(f"Showing {visible} of {total} entries")

    def _get_date_from(self) -> Optional[datetime]:
        """Get the 'from' date filter value."""
        dt = self._date_from.dateTime()
        if dt.isNull() or not dt.isValid():
            return None
        return dt.toPyDateTime()

    def _get_date_to(self) -> Optional[datetime]:
        """Get the 'to' date filter value."""
        dt = self._date_to.dateTime()
        if not dt.isValid():
            return None
        return dt.toPyDateTime()


# ── Formatters ──

def _format_value(value: Any, indent: int = 0) -> str:
    """Format a value for display in the detail pane."""
    prefix = " " * indent

    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}  {k}:")
                lines.append(_format_value(v, indent + 4))
            else:
                val_str = str(v)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                lines.append(f"{prefix}  {k}: {val_str}")
        return "\n".join(lines)

    elif isinstance(value, list):
        lines = []
        for i, item in enumerate(value):
            lines.append(f"{prefix}  [{i}] {_format_value(item, indent + 4).lstrip()}")
        return "\n".join(lines)

    else:
        val_str = str(value)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        return f"{prefix}{val_str}"
