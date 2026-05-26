"""
Rekarisk Core — Audit Trail Engine.

Version log and change tracking for all project operations.
Provides immutable, timestamped records of every significant action
performed within a project—create, modify, delete, run, export, import.

Each AuditEntry records the action, affected module, human-readable
description, before/after data snapshots, and a checksum for integrity
verification. The AuditTrail aggregates these entries and supports
filtering, diffing, and JSON import/export.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Audit Actions ──

class AuditAction:
    """Well-known audit action types."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RUN = "run"
    EXPORT = "export"
    IMPORT = "import"
    CLONE = "clone"
    RESTORE = "restore"

    ALL = {CREATE, MODIFY, DELETE, RUN, EXPORT, IMPORT, CLONE, RESTORE}


# ── Audit Entry ──

@dataclass
class AuditEntry:
    """A single immutable record in the audit trail.

    Attributes:
        timestamp: UTC datetime when the action occurred.
        action: One of AuditAction values ('create', 'modify', 'delete', 'run', etc.).
        module: Which module was affected (e.g. 'dispersion', 'fire', 'project').
        description: Human-readable summary of what happened.
        user: Identity of the user/agent who performed the action.
        details: Arbitrary key-value context (before/after snapshots, parameters, etc.).
        checksum: SHA-256 hash of the serialized details for integrity verification.
        entry_id: Auto-generated unique identifier for this entry.
    """
    timestamp: datetime
    action: str
    module: str
    description: str
    user: str = "default"
    details: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    entry_id: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
        if not self.entry_id:
            self.entry_id = self._generate_id()

    def _compute_checksum(self) -> str:
        """SHA-256 of the (action, module, serialized details)."""
        payload = json.dumps({
            "action": self.action,
            "module": self.module,
            "details": self.details,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _generate_id(self) -> str:
        """Generate a short unique ID from timestamp + checksum prefix."""
        ts = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return f"audit-{ts}-{self.checksum[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "module": self.module,
            "description": self.description,
            "user": self.user,
            "details": self.details,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEntry:
        """Deserialize from a dict."""
        ts = data.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        return cls(
            timestamp=timestamp,
            action=data.get("action", "modify"),
            module=data.get("module", "unknown"),
            description=data.get("description", ""),
            user=data.get("user", "default"),
            details=data.get("details", {}),
            checksum=data.get("checksum", ""),
            entry_id=data.get("entry_id", ""),
        )

    def verify_checksum(self) -> bool:
        """Verify that the stored checksum matches recomputed value."""
        return self.checksum == self._compute_checksum()


# ── Audit Trail ──

class AuditTrail:
    """Manages a collection of AuditEntry records for a project.

    Provides logging, filtering, diffing, and JSON serialization.
    Thread-safe for append-only operations but callers should
    serialize if iterating while appending is possible.
    """

    _MAX_ENTRIES = 10_000  # safety cap to avoid memory blowout

    def __init__(self):
        self._entries: List[AuditEntry] = []

    # ── Logging ──

    def log(
        self,
        action: str,
        module: str,
        description: str,
        user: str = "default",
        details: Dict[str, Any] | None = None,
        before: Any = None,
        after: Any = None,
    ) -> AuditEntry:
        """Record a new audit entry.

        Args:
            action: AuditAction constant (e.g. 'create', 'modify').
            module: Affected module name.
            description: Human-readable description.
            user: Identity of the actor.
            details: Arbitrary context dict.
            before: Convenience — stored as details['before'].
            after: Convenience — stored as details['after'].

        Returns:
            The created AuditEntry.
        """
        full_details = details.copy() if details else {}
        if before is not None:
            full_details["before"] = _make_serializable(before)
        if after is not None:
            full_details["after"] = _make_serializable(after)

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            action=action,
            module=module,
            description=description,
            user=user,
            details=full_details,
        )
        self._entries.append(entry)

        # Prune if over max
        if len(self._entries) > self._MAX_ENTRIES:
            self._entries = self._entries[-self._MAX_ENTRIES:]

        return entry

    def log_entry(self, entry: AuditEntry) -> None:
        """Append a pre-constructed AuditEntry."""
        self._entries.append(entry)
        if len(self._entries) > self._MAX_ENTRIES:
            self._entries = self._entries[-self._MAX_ENTRIES:]

    # ── Querying ──

    def get_history(
        self,
        module: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> List[AuditEntry]:
        """Retrieve filtered audit entries.

        Args:
            module: Filter by affected module (None = all).
            action: Filter by action type (None = all).
            since: Only return entries after this datetime.
            limit: Max number of entries to return (most recent first).

        Returns:
            List of AuditEntry matching the filters, newest first.
        """
        results = reversed(self._entries)

        filtered = []
        for entry in results:
            if module is not None and entry.module != module:
                continue
            if action is not None and entry.action != action:
                continue
            if since is not None and entry.timestamp < since:
                continue
            filtered.append(entry)
            if limit is not None and len(filtered) >= limit:
                break

        return filtered

    def get_last_run(self, module: str) -> Optional[AuditEntry]:
        """Get the most recent 'run' entry for a given module.

        Args:
            module: Module name to look up.

        Returns:
            The most recent AuditEntry with action='run' for this module,
            or None if not found.
        """
        for entry in reversed(self._entries):
            if entry.module == module and entry.action == AuditAction.RUN:
                return entry
        return None

    def get_entry_by_id(self, entry_id: str) -> Optional[AuditEntry]:
        """Find an entry by its unique ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    # ── Diffing ──

    @staticmethod
    def diff(entry1: AuditEntry, entry2: AuditEntry) -> Dict[str, Any]:
        """Compute what changed between two audit entries.

        Compares details dictionaries key-by-key. Returns a dict with:
            - 'added': keys present in entry2 but not entry1
            - 'removed': keys present in entry1 but not entry2
            - 'changed': keys with different values between the two
            - 'unchanged': keys with identical values

        Returns:
            Dict describing the differences.
        """
        d1 = entry1.details
        d2 = entry2.details
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())

        result = {
            "added": {k: d2[k] for k in (keys2 - keys1)},
            "removed": {k: d1[k] for k in (keys1 - keys2)},
            "changed": {},
            "unchanged": {},
        }

        for k in keys1 & keys2:
            if d1[k] != d2[k]:
                result["changed"][k] = {"from": d1[k], "to": d2[k]}
            else:
                result["unchanged"][k] = d1[k]

        return result

    # ── Batch Operations ──

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def entries(self) -> List[AuditEntry]:
        """Return all entries (oldest first)."""
        return list(self._entries)

    def clear(self) -> None:
        """Remove all audit entries."""
        self._entries.clear()

    # ── Serialization ──

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize all entries to a list of dicts."""
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> AuditTrail:
        """Deserialize from a list of dicts."""
        trail = cls()
        for item in data:
            trail._entries.append(AuditEntry.from_dict(item))
        return trail

    def export_log(self, path: str | Path) -> None:
        """Export the entire audit trail to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "format": "rekarisk-audit-1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": self.to_dict_list(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def import_log(self, path: str | Path) -> int:
        """Import audit entries from a JSON file (appends to existing).

        Args:
            path: Input file path.

        Returns:
            Number of entries imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        for item in entries:
            entry = AuditEntry.from_dict(item)
            self._entries.append(entry)

        return len(entries)

    def verify_all(self) -> Dict[str, Any]:
        """Verify checksum integrity of all entries.

        Returns:
            Dict with 'total', 'valid', 'invalid' counts and list of
            invalid entry IDs.
        """
        invalid_ids = []
        for entry in self._entries:
            if not entry.verify_checksum():
                invalid_ids.append(entry.entry_id)

        return {
            "total": len(self._entries),
            "valid": len(self._entries) - len(invalid_ids),
            "invalid": len(invalid_ids),
            "invalid_ids": invalid_ids,
        }


# ── Helpers ──

def _make_serializable(obj: Any) -> Any:
    """Attempt to convert an object to a JSON-serializable form.

    Handles dataclasses, datetime, and basic types; falls back to str().
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__}>"
