"""
Rekarisk Text Export — CSV, JSON, and TXT export.

Provides:
    - CSV export: tabular results (distance, concentration, etc.)
    - JSON export: complete project data + results (round-trip capable)
    - TXT export: human-readable summary report
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ══════════════════════════════════════════════════════════════════════════════
# CSV Export
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    delimiter: str = ",",
) -> str:
    """Export results tables as CSV.

    Creates one CSV file per scenario that has table_rows data.
    If only one scenario with data, saves to exactly output_path.
    If multiple, saves as output_path_stem_N.csv.

    Parameters
    ----------
    results : list[dict]
        List of result dicts with optional "table_headers" and "table_rows".
    output_path : str or Path
        Target path. If multiple scenarios, uses this as base name.
    delimiter : str
        CSV delimiter (default comma).

    Returns
    -------
    str
        The output path (or path of first file if multiple).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_with_tables = [r for r in results if r.get("table_rows")]

    if not results_with_tables:
        # Write a minimal CSV with just summary data
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(["scenario", "type", "key", "value"])
            for res in results:
                summary = res.get("summary", {})
                for k, v in summary.items():
                    writer.writerow([res.get("name", ""), res.get("type", ""), k, v])
        return str(output_path)

    if len(results_with_tables) == 1:
        paths = [_write_single_csv(results_with_tables[0], output_path, delimiter)]
    else:
        stem = output_path.stem
        suffix = output_path.suffix or ".csv"
        paths = []
        for i, res in enumerate(results_with_tables, 1):
            p = output_path.parent / f"{stem}_{i}{suffix}"
            _write_single_csv(res, p, delimiter)
            paths.append(p)

    return str(paths[0])


def _write_single_csv(result: Dict[str, Any], path: Path, delimiter: str) -> Path:
    """Write a single result to CSV. Returns the path."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)

        # Header info
        writer.writerow([f"# Scenario: {result.get('name', 'Unnamed')}"])
        writer.writerow([f"# Type: {result.get('type', 'N/A')}"])
        writer.writerow([])

        # Table data
        headers = result.get("table_headers", [])
        rows = result.get("table_rows", [])

        if headers:
            writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# JSON Export
# ══════════════════════════════════════════════════════════════════════════════

def export_json(
    project_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    """Export complete project data and results as JSON.

    The output can be loaded back for inspection or archival.
    NumPy types are converted to native Python types.

    Parameters
    ----------
    project_data : dict
        Full project metadata.
    results : list[dict]
        All scenario results.
    output_path : str or Path
        Where to save the .json file.

    Returns
    -------
    str
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make a copy so we don't mutate originals
    export_data = {
        "format": "rekarisk-export-v1",
        "exported_at": datetime.now().isoformat(),
        "project": _sanitize_for_json(project_data),
        "results": [_sanitize_for_json(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    return str(output_path)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-serializable types (numpy, etc.) to JSON-safe types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()
                if not k.startswith("_")}  # Skip private-like keys
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, (Path,)):
        return str(obj)
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# TXT Export
# ══════════════════════════════════════════════════════════════════════════════

def export_summary_text(
    project_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    """Export a human-readable summary text file.

    Parameters
    ----------
    project_data : dict
        Project metadata.
    results : list[dict]
        All scenario results.
    output_path : str or Path
        Where to save the .txt file.

    Returns
    -------
    str
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("REKARISK CONSEQUENCE ANALYSIS REPORT")
    lines.append(sep)
    lines.append(f"Project:      {project_data.get('name', 'Untitled')}")
    lines.append(f"Description:  {project_data.get('description', 'N/A')}")
    lines.append(f"Date:         {datetime.now().strftime('%d %B %Y')}")
    lines.append(f"Exported:     {datetime.now().isoformat()}")
    lines.append(sep)
    lines.append("")

    # Scenarios overview
    lines.append(f"TOTAL SCENARIOS: {len(results)}")
    lines.append("")

    for i, res in enumerate(results, 1):
        lines.append(f"── SCENARIO {i}: {res.get('name', 'Unnamed')} ──")
        lines.append(f"  Type:       {res.get('type', 'N/A')}")

        # Summary
        summary = res.get("summary", {})
        if summary:
            lines.append("  Summary:")
            for k, v in summary.items():
                lines.append(f"    {k}: {v}")

        # Thresholds
        thresholds = res.get("thresholds", {})
        if thresholds:
            lines.append("  Threshold Distances:")
            for label, dist in thresholds.items():
                val = f"{dist:.1f} m" if isinstance(dist, (int, float)) else str(dist)
                lines.append(f"    {label}: {val}")

        # Risk
        risk_level = res.get("risk_level")
        if risk_level:
            lines.append(f"  Risk Level: {risk_level}")

        ir_thresholds = res.get("ir_thresholds", {})
        if ir_thresholds:
            lines.append("  Individual Risk Contours:")
            for label, dist in ir_thresholds.items():
                val = f"{dist:.1f} m" if isinstance(dist, (int, float)) else str(dist)
                lines.append(f"    {label}: {val}")

        lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(output_path)
