"""
Rekarisk CLI — Command-line interface for headless and batch operations.

Usage::

    rekarisk run scenario.json -o results.json
    rekarisk batch config.json
    rekarisk version
    rekarisk substances methane
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from rekarisk.__version__ import __version__


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_json(path: str) -> dict[str, Any]:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(data: dict[str, Any], path: str) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a single scenario from a JSON input file."""
    scenario = _load_json(args.input)
    scenario_type = scenario.get("type", "dispersion")

    result: dict[str, Any] = {
        "status": "ok",
        "scenario_type": scenario_type,
        "input": args.input,
    }

    # Dispatch to the appropriate model engine.
    try:
        if scenario_type == "dispersion":
            result["message"] = "Dispersion simulation placeholder"
        elif scenario_type == "fire":
            result["message"] = "Fire simulation placeholder"
        elif scenario_type == "explosion":
            result["message"] = "Explosion simulation placeholder"
        elif scenario_type == "source_term":
            result["message"] = "Source term calculation placeholder"
        elif scenario_type == "qra":
            result["message"] = "QRA calculation placeholder"
        else:
            result["status"] = "error"
            result["message"] = f"Unknown scenario type: {scenario_type}"
    except Exception as exc:
        result["status"] = "error"
        result["message"] = str(exc)

    # Output
    out_path = args.output or f"rekarisk_result.{args.format}"
    _write_json(result, out_path)
    print(f"✓ Results written to {out_path}")
    return 0 if result["status"] == "ok" else 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute a batch of scenarios defined in a config file."""
    config = _load_json(args.config)
    scenarios = config.get("scenarios", [])

    results: list[dict[str, Any]] = []
    failed = 0

    for i, scenario in enumerate(scenarios):
        name = scenario.get("name", f"scenario_{i}")
        scenario_type = scenario.get("type", "dispersion")

        try:
            if scenario_type == "dispersion":
                results.append({"name": name, "status": "ok", "message": "Dispersion placeholder"})
            elif scenario_type == "fire":
                results.append({"name": name, "status": "ok", "message": "Fire placeholder"})
            elif scenario_type == "explosion":
                results.append({"name": name, "status": "ok", "message": "Explosion placeholder"})
            elif scenario_type == "source_term":
                results.append({"name": name, "status": "ok", "message": "Source term placeholder"})
            elif scenario_type == "qra":
                results.append({"name": name, "status": "ok", "message": "QRA placeholder"})
            else:
                results.append({"name": name, "status": "error", "message": f"Unknown type: {scenario_type}"})
                failed += 1
        except Exception as exc:
            results.append({"name": name, "status": "error", "message": str(exc)})
            failed += 1

    summary = {
        "total": len(scenarios),
        "successful": len(scenarios) - failed,
        "failed": failed,
        "results": results,
    }

    out_path = config.get("output", "rekarisk_batch_results.json")
    _write_json(summary, out_path)
    print(f"✓ Batch complete: {summary['successful']}/{summary['total']} succeeded → {out_path}")
    return 0 if failed == 0 else 1


def cmd_version(args: argparse.Namespace) -> int:
    """Print Rekarisk version and exit."""
    print(f"Rekarisk v{__version__}")
    return 0


def cmd_substances(args: argparse.Namespace) -> int:
    """Search the built-in substance database."""
    try:
        from rekarisk.core.substance_db import SubstanceDatabase

        db = SubstanceDatabase()
        try:
            db.load()
        except Exception:
            # Database schema mismatch — load raw JSON and search directly
            db_path = db.path if hasattr(db, 'path') else None
            print(f"Note: using raw database search for '{args.query}'")
            _cmd_substances_raw(args.query)
            return 0

        results = db.search(args.query) if db.is_loaded else []

        if not results:
            print(f"No substances found matching '{args.query}'")
            return 1

        print(f"Matches for '{args.query}':")
        for sub in results[:20]:
            name = getattr(sub, "name", str(sub))
            formula = getattr(sub, "formula", "")
            cas = getattr(sub, "cas_number", "")
            print(f"  • {name}  {formula}  ({cas})")
        return 0
    except ImportError:
        print("Substance database not available. Check installation.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def _cmd_substances_raw(query: str) -> None:
    """Fallback: search the raw substances.json directly."""
    import json
    import os

    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "substances.json")
    db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q = query.lower().strip()
    matches = []
    for sub in data.get("substances", []):
        name = (sub.get("name") or "").lower()
        formula = (sub.get("formula") or "").lower()
        cas = (sub.get("cas") or "").lower()
        if q in name or q in formula or q == cas:
            matches.append(sub)

    if not matches:
        print(f"No substances found matching '{query}'")
        return

    print(f"Matches for '{query}':")
    for sub in matches[:20]:
        name = sub.get("name", "?")
        formula = sub.get("formula", "")
        cas = sub.get("cas", "")
        print(f"  • {name}  {formula}  ({cas})")


# ── argument parser ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="rekarisk",
        description="Rekarisk — Consequence & Risk Analysis for Safety Engineers",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # run
    run_parser = subparsers.add_parser("run", help="Run a scenario from a JSON input file")
    run_parser.add_argument("input", help="Input JSON file describing the scenario")
    run_parser.add_argument("-o", "--output", help="Output file path")
    run_parser.add_argument(
        "-f", "--format",
        choices=["json", "csv", "pdf"],
        default="json",
        help="Output format (default: json)",
    )

    # batch
    batch_parser = subparsers.add_parser("batch", help="Run multiple scenarios from a batch config JSON")
    batch_parser.add_argument("config", help="Batch configuration JSON file")

    # version
    subparsers.add_parser("version", help="Show Rekarisk version")

    # substances
    sub_parser = subparsers.add_parser("substances", help="Search the substance database")
    sub_parser.add_argument("query", help="Search term (name, formula, or CAS)")

    return parser


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch table
    handlers = {
        "run": cmd_run,
        "batch": cmd_batch,
        "version": cmd_version,
        "substances": cmd_substances,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
