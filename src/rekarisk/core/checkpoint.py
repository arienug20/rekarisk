"""
Rekarisk Core — Checkpoint / Rollback System.

Provides automatic and manual project state snapshots so users can
undo large changes or recover from errors. Checkpoints are stored
as compact JSON files in a per-project directory under
~/.rekarisk/checkpoints/.

Key behaviors:
- Auto-checkpoint before risky operations (delete, batch run, import).
- Manual checkpoint via Ctrl+Shift+S or menu action.
- Maximum 20 checkpoints per project; oldest is pruned on overflow.
- Restore replaces the in-memory project state from any checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit_trail import AuditAction


# ── Constants ──

CHECKPOINT_ROOT = Path.home() / ".rekarisk" / "checkpoints"
MAX_CHECKPOINTS = 20
CHECKPOINT_FILE = "checkpoint.json"
CHECKPOINT_DATA_FILE = "project_data.json"


# ── Checkpoint Manager ──

class Checkpoint:
    """Manages project checkpoints for a single project.

    Each checkpoint is a directory containing:
      - checkpoint.json : metadata (id, label, timestamp, size_bytes)
      - project_data.json : full snapshot of project state

    Usage::

        mgr = Checkpoint("my_project_123")
        cid = mgr.create(project_data, "Before scaling analysis")
        ...
        restored = mgr.restore(cid)
    """

    def __init__(self, project_id: str):
        """Initialize checkpoint manager for a project.

        Args:
            project_id: Unique identifier for the project (usually derived
                        from the project name or file path hash).
        """
        if not project_id.strip():
            raise ValueError("project_id must be non-empty")

        self._project_id = project_id
        self._base_dir = CHECKPOINT_ROOT / project_id
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──

    def create(self, project_data: Dict[str, Any], label: str = "") -> str:
        """Create a new checkpoint from the given project data.

        Args:
            project_data: Full project state dict to snapshot.
            label: Human-readable label (e.g. "Before batch run").

        Returns:
            The checkpoint ID (a short hash string).
        """
        checkpoint_id = self._make_id(label)
        check_dir = self._base_dir / checkpoint_id
        check_dir.mkdir(parents=True, exist_ok=True)

        # Serialize project data
        data_json = json.dumps(project_data, indent=2, ensure_ascii=False, default=str)
        data_bytes = len(data_json.encode("utf-8"))

        # Write project data
        data_path = check_dir / CHECKPOINT_DATA_FILE
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(data_json)

        # Write checkpoint metadata
        meta = {
            "id": checkpoint_id,
            "label": label or f"Checkpoint {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_bytes": data_bytes,
            "project_id": self._project_id,
            "scenario_count": len(project_data.get("scenarios", [])),
        }
        meta_path = check_dir / CHECKPOINT_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Prune old checkpoints if over max
        self._prune()

        return checkpoint_id

    def restore(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore project data from a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to restore.

        Returns:
            The project data dict as it was when snapshotted.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        check_dir = self._base_dir / checkpoint_id
        data_path = check_dir / CHECKPOINT_DATA_FILE

        if not data_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_id}' not found for project "
                f"'{self._project_id}'"
            )

        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to delete.

        Returns:
            True if the checkpoint existed and was deleted, False otherwise.
        """
        check_dir = self._base_dir / checkpoint_id
        if check_dir.exists():
            shutil.rmtree(check_dir)
            return True
        return False

    def delete_all(self) -> int:
        """Delete all checkpoints for this project.

        Returns:
            Number of checkpoints deleted.
        """
        count = len(self.list_checkpoints())
        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
        return count

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints for this project, newest first.

        Returns:
            List of checkpoint metadata dicts with keys:
            id, label, timestamp, size_bytes, project_id, scenario_count.
        """
        checkpoints = []
        if not self._base_dir.exists():
            return checkpoints

        for entry in sorted(self._base_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            meta_path = entry / CHECKPOINT_FILE
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    checkpoints.append(meta)
                except (json.JSONDecodeError, OSError):
                    # Corrupt checkpoint — skip
                    continue

        # Sort newest first
        checkpoints.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
        return checkpoints

    def auto_checkpoint(self, project_data: Dict[str, Any],
                        module: str) -> Optional[str]:
        """Automatically create a checkpoint before a risky operation.

        Intended to be called before operations like:
        - Batch runs
        - Deleting scenarios
        - Importing data
        - Mass-editing

        Args:
            project_data: Current project state.
            module: Name of the module triggering the checkpoint (for the label).

        Returns:
            The checkpoint ID, or None if creation failed.
        """
        label = f"Auto: before {module} operation"
        try:
            # Throttle: skip if last auto-checkpoint was within 10 seconds
            cps = self.list_checkpoints()
            if cps:
                last_ts = cps[0].get("timestamp", "")
                if last_ts:
                    try:
                        last_dt = datetime.fromisoformat(last_ts)
                        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        if elapsed < 10 and "Auto:" in cps[0].get("label", ""):
                            return None  # skip duplicate auto-checkpoints
                    except (ValueError, TypeError):
                        pass

            return self.create(project_data, label)
        except OSError:
            return None

    def get_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID.

        Returns:
            Metadata dict or None if not found.
        """
        meta_path = self._base_dir / checkpoint_id / CHECKPOINT_FILE
        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the latest (most recent) checkpoint metadata.

        Returns:
            Metadata dict or None if no checkpoints exist.
        """
        cps = self.list_checkpoints()
        return cps[0] if cps else None

    # ── Helpers ──

    def _make_id(self, label: str) -> str:
        """Generate a unique checkpoint ID."""
        raw = f"{self._project_id}-{label}-{time.time_ns()}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"ck-{ts}-{h}"

    def _prune(self) -> None:
        """Remove oldest checkpoints if count exceeds MAX_CHECKPOINTS."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= MAX_CHECKPOINTS:
            return

        # Remove oldest (keep newest MAX_CHECKPOINTS)
        to_remove = checkpoints[MAX_CHECKPOINTS:]
        for cp in to_remove:
            self.delete(cp["id"])

    # ── Stats ──

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics for this project's checkpoints."""
        cps = self.list_checkpoints()
        total_size = sum(c.get("size_bytes", 0) for c in cps)
        return {
            "project_id": self._project_id,
            "count": len(cps),
            "max_allowed": MAX_CHECKPOINTS,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_dir": str(self._base_dir),
        }

    def __repr__(self) -> str:
        return (f"Checkpoint(project_id={self._project_id!r}, "
                f"count={len(self.list_checkpoints())})")


# ── Module-level helpers ──

def get_project_id_from_path(path: str | Path) -> str:
    """Derive a stable project ID from a file path.

    Uses the SHA-256 of the canonical path to produce a consistent
    ID that survives renames.
    """
    path = Path(path).resolve()
    h = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    return f"{path.stem}-{h}"


def get_project_id_from_name(name: str) -> str:
    """Derive a project ID from a project name (for unsaved projects)."""
    sanitized = "".join(c for c in name if c.isalnum() or c in "_- ").strip()
    if not sanitized:
        sanitized = "untitled"
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{sanitized}-{h}"


def get_total_checkpoint_size() -> Dict[str, Any]:
    """Get aggregate checkpoint storage stats across all projects."""
    total_bytes = 0
    project_count = 0
    if CHECKPOINT_ROOT.exists():
        for proj_dir in CHECKPOINT_ROOT.iterdir():
            if proj_dir.is_dir():
                project_count += 1
                for cp_dir in proj_dir.iterdir():
                    if cp_dir.is_dir():
                        data_path = cp_dir / CHECKPOINT_DATA_FILE
                        if data_path.exists():
                            total_bytes += data_path.stat().st_size

    return {
        "project_count": project_count,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "checkpoint_root": str(CHECKPOINT_ROOT),
    }


def purge_all_checkpoints() -> int:
    """Delete all checkpoints across all projects. Use with caution!

    Returns:
        Number of project directories removed.
    """
    count = 0
    if CHECKPOINT_ROOT.exists():
        for proj_dir in CHECKPOINT_ROOT.iterdir():
            if proj_dir.is_dir():
                shutil.rmtree(proj_dir)
                count += 1
    return count
