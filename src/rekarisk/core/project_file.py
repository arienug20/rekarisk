"""
Rekarisk Core — Project File Format (.caproj).

A portable, self-contained ZIP archive that holds the complete
state of a Rekarisk project: metadata, scenarios, weather data,
terrain data, settings, audit trail, and result artifacts.

Format structure inside the ZIP:
  project.json          — Main project data (metadata, scenarios, settings)
  audit.json            — Embedded audit trail entries
  results/              — Result data files (CSV, JSON, NumPy .npy)
  plots/                — Saved plot images (PNG, SVG)
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .audit_trail import AuditTrail, AuditEntry, AuditAction


# Current project format version
FORMAT_VERSION = "1.0"
FILE_EXTENSION = ".caproj"
FILE_FILTER = f"Rekarisk Project Files (*.caproj);;All Files (*)"

# Max ZIP member size (100 MB) for safety
MAX_MEMBER_SIZE = 100 * 1024 * 1024


# ── Project Metadata ──

@dataclass
class ProjectMetadata:
    """Lightweight project identity and version info."""
    name: str = "Untitled"
    description: str = ""
    author: str = ""
    created: str = ""
    modified: str = ""
    version: str = FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created": self.created or datetime.now(timezone.utc).isoformat(),
            "modified": self.modified or datetime.now(timezone.utc).isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectMetadata:
        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created=data.get("created", ""),
            modified=data.get("modified", ""),
            version=data.get("version", FORMAT_VERSION),
        )


# ── Project Data Container ──

@dataclass
class ProjectData:
    """Complete project state ready for serialization."""
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    weather_data: Dict[str, Any] = field(default_factory=dict)
    terrain_data: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    audit_entries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "metadata": self.metadata.to_dict(),
            "scenarios": self.scenarios,
            "weather_data": self.weather_data,
            "terrain_data": self.terrain_data,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectData:
        return cls(
            metadata=ProjectMetadata.from_dict(data.get("metadata", {})),
            scenarios=data.get("scenarios", []),
            weather_data=data.get("weather_data", {}),
            terrain_data=data.get("terrain_data", {}),
            settings=data.get("settings", {}),
            audit_entries=[],  # loaded separately from audit.json
        )


# ── Project File ──

class ProjectFile:
    """Handler for .caproj ZIP-based project files.

    Reads and writes a self-contained project archive that bundles
    the main JSON manifest, audit trail, result artifacts, and plots.

    Usage::

        proj = ProjectFile()
        proj.metadata.name = "My Analysis"
        proj.scenarios.append({...})
        proj.save("/path/to/project.caproj")

        loaded = ProjectFile.load("/path/to/project.caproj")
    """

    def __init__(self):
        self._metadata = ProjectMetadata()
        self._scenarios: List[Dict[str, Any]] = []
        self._weather_data: Dict[str, Any] = {}
        self._terrain_data: Dict[str, Any] = {}
        self._settings: Dict[str, Any] = {}
        self._audit_trail = AuditTrail()
        self._results_files: Dict[str, bytes] = {}    # filename → binary content
        self._plot_files: Dict[str, bytes] = {}       # filename → binary content

    # ── Properties ──

    @property
    def metadata(self) -> ProjectMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: ProjectMetadata):
        self._metadata = value

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return self._scenarios

    @scenarios.setter
    def scenarios(self, value: List[Dict[str, Any]]):
        self._scenarios = value

    @property
    def weather_data(self) -> Dict[str, Any]:
        return self._weather_data

    @weather_data.setter
    def weather_data(self, value: Dict[str, Any]):
        self._weather_data = value

    @property
    def terrain_data(self) -> Dict[str, Any]:
        return self._terrain_data

    @terrain_data.setter
    def terrain_data(self, value: Dict[str, Any]):
        self._terrain_data = value

    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings

    @settings.setter
    def settings(self, value: Dict[str, Any]):
        self._settings = value

    @property
    def audit_trail(self) -> AuditTrail:
        return self._audit_trail

    @property
    def result_files(self) -> Dict[str, bytes]:
        return self._results_files

    @property
    def plot_files(self) -> Dict[str, bytes]:
        return self._plot_files

    # ── Project-wide Data ──

    def get_project_data(self) -> Dict[str, Any]:
        """Return the full project state as a dict (for checkpointing, etc.)."""
        return {
            "metadata": self._metadata.to_dict(),
            "scenarios": deepcopy(self._scenarios),
            "weather_data": deepcopy(self._weather_data),
            "terrain_data": deepcopy(self._terrain_data),
            "settings": deepcopy(self._settings),
            "audit_entries": self._audit_trail.to_dict_list(),
            "result_files": {k: len(v) for k, v in self._results_files.items()},
            "plot_files": {k: len(v) for k, v in self._plot_files.items()},
        }

    def set_project_data(self, data: Dict[str, Any]) -> None:
        """Restore project state from a dict."""
        self._metadata = ProjectMetadata.from_dict(data.get("metadata", {}))
        self._scenarios = deepcopy(data.get("scenarios", []))
        self._weather_data = deepcopy(data.get("weather_data", {}))
        self._terrain_data = deepcopy(data.get("terrain_data", {}))
        self._settings = deepcopy(data.get("settings", {}))

        audit_entries = data.get("audit_entries", [])
        self._audit_trail = AuditTrail.from_dict_list(audit_entries)

    # ── Save / Load ──

    def save(self, path: str | Path, progress_callback=None) -> str:
        """Save the project to a .caproj ZIP file.

        Args:
            path: Output file path (should end in .caproj).
            progress_callback: Optional callable(step, total) for UI updates.

        Returns:
            The resolved output path as a string.

        Raises:
            OSError: If the file cannot be written.
            ValueError: If the path doesn't have .caproj extension.
        """
        path = Path(path)

        # Update metadata timestamps
        now = datetime.now(timezone.utc).isoformat()
        if not self._metadata.created:
            self._metadata.created = now
        self._metadata.modified = now

        # Build project JSON
        project_dict = {
            "format_version": FORMAT_VERSION,
            "metadata": self._metadata.to_dict(),
            "scenarios": self._scenarios,
            "weather_data": self._weather_data,
            "terrain_data": self._terrain_data,
            "settings": self._settings,
        }

        # Build audit JSON
        audit_dict = {
            "format": "rekarisk-audit-1.0",
            "entry_count": len(self._audit_trail),
            "entries": self._audit_trail.to_dict_list(),
        }

        steps = 2 + len(self._results_files) + len(self._plot_files)
        step = 0

        path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write project.json
            zf.writestr("project.json",
                        json.dumps(project_dict, indent=2, ensure_ascii=False))
            step += 1
            if progress_callback:
                progress_callback(step, steps)

            # Write audit.json
            zf.writestr("audit.json",
                        json.dumps(audit_dict, indent=2, ensure_ascii=False,
                                   default=str))
            step += 1
            if progress_callback:
                progress_callback(step, steps)

            # Write result files
            for filename, content in self._results_files.items():
                arcname = f"results/{filename}"
                zf.writestr(arcname, content)
                step += 1
                if progress_callback:
                    progress_callback(step, steps)

            # Write plot files
            for filename, content in self._plot_files.items():
                arcname = f"plots/{filename}"
                zf.writestr(arcname, content)
                step += 1
                if progress_callback:
                    progress_callback(step, steps)

        # Log the save action
        self._audit_trail.log(
            action=AuditAction.EXPORT,
            module="project",
            description=f"Project saved to {path.name}",
            details={
                "path": str(path),
                "scenario_count": len(self._scenarios),
                "file_size": path.stat().st_size if path.exists() else 0,
            },
        )

        return str(path)

    @classmethod
    def load(cls, path: str | Path, progress_callback=None) -> ProjectFile:
        """Load a project from a .caproj ZIP file.

        Args:
            path: Path to the .caproj file.
            progress_callback: Optional callable(step, total) for UI updates.

        Returns:
            A populated ProjectFile instance.

        Raises:
            FileNotFoundError: If path does not exist.
            zipfile.BadZipFile: If the file is not a valid ZIP.
            ValueError: If the archive structure is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        project = cls()

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            # Validate basic structure
            if "project.json" not in names:
                raise ValueError("Invalid .caproj: missing project.json")

            steps = len(names)
            step = 0

            # Load project.json
            with zf.open("project.json") as f:
                raw = f.read(MAX_MEMBER_SIZE).decode("utf-8")
                project_dict = json.loads(raw)

            project._metadata = ProjectMetadata.from_dict(
                project_dict.get("metadata", {}))
            project._scenarios = project_dict.get("scenarios", [])
            project._weather_data = project_dict.get("weather_data", {})
            project._terrain_data = project_dict.get("terrain_data", {})
            project._settings = project_dict.get("settings", {})

            step += 1
            if progress_callback:
                progress_callback(step, steps)

            # Load audit.json (optional)
            if "audit.json" in names:
                with zf.open("audit.json") as f:
                    raw = f.read(MAX_MEMBER_SIZE).decode("utf-8")
                    audit_dict = json.loads(raw)
                project._audit_trail = AuditTrail.from_dict_list(
                    audit_dict.get("entries", []))
            step += 1
            if progress_callback:
                progress_callback(step, steps)

            # Load result/ and plot/ files
            for name in names:
                if name in ("project.json", "audit.json"):
                    continue
                info = zf.getinfo(name)
                if info.file_size > MAX_MEMBER_SIZE:
                    continue  # skip oversized members
                with zf.open(name) as f:
                    content = f.read()

                basename = name.split("/", 1)[-1]
                if name.startswith("results/") and basename:
                    project._results_files[basename] = content
                elif name.startswith("plots/") and basename:
                    project._plot_files[basename] = content

                step += 1
                if progress_callback:
                    progress_callback(step, steps)

        # Log the load action
        project._audit_trail.log(
            action=AuditAction.IMPORT,
            module="project",
            description=f"Project loaded from {path.name}",
            details={
                "path": str(path),
                "scenario_count": len(project._scenarios),
                "file_size": path.stat().st_size,
            },
        )

        return project

    # ── Result / Plot Management ──

    def add_result_file(self, name: str, content: bytes) -> None:
        """Add or replace a result file in the project.

        Args:
            name: Filename within results/ (e.g. 'dispersion_1.csv').
            content: Binary file content.
        """
        self._results_files[name] = content

    def add_plot_file(self, name: str, content: bytes) -> None:
        """Add or replace a plot image in the project.

        Args:
            name: Filename within plots/ (e.g. 'dispersion_concentration.png').
            content: Binary image data.
        """
        self._plot_files[name] = content

    def remove_result_file(self, name: str) -> bool:
        """Remove a result file. Returns True if it existed."""
        return self._results_files.pop(name, None) is not None

    def remove_plot_file(self, name: str) -> bool:
        """Remove a plot file. Returns True if it existed."""
        return self._plot_files.pop(name, None) is not None

    # ── Scenario Export / Import ──

    def export_scenario(self, scenario_id: int, format: str = "json") -> str:
        """Export a single scenario as a JSON string.

        Args:
            scenario_id: Index into the scenarios list.
            format: Output format ('json' only for now).

        Returns:
            JSON string of the scenario data.

        Raises:
            IndexError: If scenario_id is invalid.
        """
        if scenario_id < 0 or scenario_id >= len(self._scenarios):
            raise IndexError(f"Scenario {scenario_id} not found")

        scenario = deepcopy(self._scenarios[scenario_id])

        self._audit_trail.log(
            action=AuditAction.EXPORT,
            module="scenario",
            description=f"Exported scenario: {scenario.get('name', scenario_id)}",
            details={"scenario_id": scenario_id, "format": format},
        )

        if format.lower() == "json":
            return json.dumps(scenario, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_scenario(self, data: str | dict | Path) -> Dict[str, Any]:
        """Import a scenario from JSON string, dict, or file path.

        Args:
            data: JSON string, dict, or path to a .json file.

        Returns:
            The imported scenario dict (also appended to scenarios list).
        """
        if isinstance(data, Path) or (isinstance(data, str) and
                                       data.endswith(".json")):
            with open(data, "r", encoding="utf-8") as f:
                scenario = json.load(f)
        elif isinstance(data, str):
            scenario = json.loads(data)
        elif isinstance(data, dict):
            scenario = deepcopy(data)
        else:
            raise TypeError(f"Cannot import from {type(data).__name__}")

        # Assign an ID if missing
        if "id" not in scenario:
            scenario["id"] = len(self._scenarios)
        if "imported_at" not in scenario:
            scenario["imported_at"] = datetime.now(timezone.utc).isoformat()

        self._scenarios.append(scenario)

        self._audit_trail.log(
            action=AuditAction.IMPORT,
            module="scenario",
            description=f"Imported scenario: {scenario.get('name', 'unknown')}",
            details={"scenario_id": scenario.get("id"), "name": scenario.get("name")},
        )

        return scenario

    # ── Data Sources (for UI integration) ──

    def to_main_window_data(self) -> Dict[str, Any]:
        """Convert to the dict format MainWindow's _project_data expects.

        This bridges the new ProjectFile model with the existing MainWindow
        data model.
        """
        return {
            "format_version": FORMAT_VERSION,
            "name": self._metadata.name,
            "description": self._metadata.description,
            "author": self._metadata.author,
            "created_at": self._metadata.created,
            "modified_at": self._metadata.modified,
            "scenarios": self._scenarios,
            "weather_cases": self._weather_data.get("cases", []),
            "weather_data": self._weather_data,
            "terrain_data": self._terrain_data,
            "substances": self._settings.get("substances", []),
            "settings": self._settings,
            "results": [],
            "reports": [],
        }

    def from_main_window_data(self, data: Dict[str, Any]) -> None:
        """Populate from the dict format MainWindow's _project_data uses."""
        self._metadata = ProjectMetadata(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created=data.get("created_at", ""),
            modified=data.get("modified_at", ""),
            version=data.get("format_version", FORMAT_VERSION),
        )
        self._scenarios = data.get("scenarios", [])
        self._weather_data = data.get("weather_data", {})
        # Also handle legacy 'weather_cases' key
        if not self._weather_data and data.get("weather_cases"):
            self._weather_data = {"cases": data.get("weather_cases", [])}
        self._terrain_data = data.get("terrain_data", {})
        self._settings = data.get("settings", {})

    # ── Validation ──

    @staticmethod
    def validate(data: Dict[str, Any]) -> List[str]:
        """Check a project data dict for structural integrity.

        Accepts either format:
          - Full project.json format (has 'metadata' key)
          - Main window data format (flat, has 'name' at top level)

        Returns:
            List of validation error messages. Empty = valid.
        """
        errors = []

        # Detect format: project.json has 'metadata' key; main_window data does not
        has_metadata_key = "metadata" in data

        if has_metadata_key:
            # project.json format
            meta = data.get("metadata", {})
            if not isinstance(meta, dict):
                errors.append("metadata must be a dict")
            else:
                if not meta.get("name"):
                    errors.append("metadata.name is required")

            scenarios = data.get("scenarios", [])
        else:
            # flat/main_window format
            if not data.get("name"):
                errors.append("name is required")

            scenarios = data.get("scenarios", [])

        # Validate scenarios (common)
        if not isinstance(scenarios, list):
            errors.append("scenarios must be a list")
        else:
            for i, s in enumerate(scenarios):
                if not isinstance(s, dict):
                    errors.append(f"scenarios[{i}] must be a dict")
                elif "type" not in s:
                    errors.append(f"scenarios[{i}] missing 'type' field")

        # Validate format version
        version = data.get("format_version", "")
        if version and version != FORMAT_VERSION:
            errors.append(
                f"format_version '{version}' differs from current '{FORMAT_VERSION}'"
            )

        return errors

    # ── Convenience ──

    def is_empty(self) -> bool:
        """Check if the project has no user data."""
        return (not self._scenarios and
                not self._weather_data.get("cases") and
                not self._terrain_data)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of the project."""
        return {
            "name": self._metadata.name,
            "description": self._metadata.description,
            "author": self._metadata.author,
            "scenario_count": len(self._scenarios),
            "weather_case_count": len(self._weather_data.get("cases", [])),
            "result_file_count": len(self._results_files),
            "plot_count": len(self._plot_files),
            "audit_entry_count": len(self._audit_trail),
            "created": self._metadata.created,
            "modified": self._metadata.modified,
        }

    def __repr__(self) -> str:
        return (f"ProjectFile(name={self._metadata.name!r}, "
                f"scenarios={len(self._scenarios)}, "
                f"audit_entries={len(self._audit_trail)})")
