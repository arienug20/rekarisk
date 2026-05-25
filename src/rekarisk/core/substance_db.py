"""
Rekarisk — Substance Database.

JSON-based substance database with search, filter, and export capabilities.
Designed for fast startup and easy editing by users.

The database is loaded from data/substances.json. Users can extend it
with custom substances via the UI.

Database format::

    {
      "version": "1.0",
      "substances": [
        { "id": "methane", "name": "Methane", ... },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .substance import Substance

# Default path relative to package root
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DB_PATH = _PACKAGE_ROOT / "data" / "substances.json"


# ══════════════════════════════════════════════════════════════════════════════
# SubstanceDatabase
# ══════════════════════════════════════════════════════════════════════════════

class SubstanceDatabase:
    """In-memory substance database with JSON persistence.

    Usage::

        db = SubstanceDatabase()
        db.load()                        # Load default database
        methane = db.get("methane")
        results = db.search("ethane")
        toxic_gases = db.filter_by_hazard("toxic")
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to JSON database file. Uses default location if None.
        """
        self._path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._substances: Dict[str, Substance] = {}
        self._by_cas: Dict[str, Substance] = {}
        self._by_un: Dict[str, Substance] = {}
        self._by_name_lower: Dict[str, Substance] = {}
        self._version: str = "1.0"
        self._loaded: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Load / Save
    # ──────────────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    def load(self, path: str | Path | None = None) -> int:
        """Load substances from JSON file.

        Args:
            path: Path to the JSON file. Uses the instance default if None.

        Returns:
            Number of substances loaded.
        """
        if path:
            self._path = Path(path)

        if not self._path.exists():
            self._loaded = True
            return 0

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._version = data.get("version", "1.0")
        substances_data = data.get("substances", [])

        self._substances.clear()
        self._by_cas.clear()
        self._by_un.clear()
        self._by_name_lower.clear()

        for item in substances_data:
            sub = Substance.from_dict(item)
            self._add_to_index(sub)

        self._loaded = True
        return len(self._substances)

    def save(self, path: str | Path | None = None) -> None:
        """Save the current database to JSON file.

        Args:
            path: Output path. Uses the current database path if None.
        """
        target = Path(path) if path else self._path

        data: Dict[str, Any] = {
            "version": self._version,
            "substances": [s.to_dict() for s in self._substances.values()],
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def reload(self) -> int:
        """Reload from the current path. Returns number of substances loaded."""
        self._substances.clear()
        self._loaded = False
        return self.load()

    # ──────────────────────────────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────────────────────────────

    def add(self, substance: Substance) -> None:
        """Add or update a substance in the database."""
        if substance.id in self._substances:
            # Remove old index entries
            old = self._substances[substance.id]
            self._remove_from_index(old)
        self._add_to_index(substance)

    def remove(self, substance_id: str) -> bool:
        """Remove a substance by ID. Returns True if found and removed."""
        sub = self._substances.pop(substance_id, None)
        if sub:
            self._remove_from_index(sub)
            return True
        return False

    def _add_to_index(self, sub: Substance) -> None:
        """Add a substance to all lookup indexes."""
        self._substances[sub.id] = sub
        if sub.cas_number:
            self._by_cas[sub.cas_number] = sub
        if sub.un_number:
            self._by_un[sub.un_number] = sub
        self._by_name_lower[sub.name.lower()] = sub

    def _remove_from_index(self, sub: Substance) -> None:
        """Remove a substance from all lookup indexes."""
        self._substances.pop(sub.id, None)
        if sub.cas_number:
            self._by_cas.pop(sub.cas_number, None)
        if sub.un_number:
            self._by_un.pop(sub.un_number, None)
        self._by_name_lower.pop(sub.name.lower(), None)

    # ──────────────────────────────────────────────────────────────────────
    # Lookup
    # ──────────────────────────────────────────────────────────────────────

    def get(self, substance_id: str) -> Substance | None:
        """Get a substance by its unique ID."""
        return self._substances.get(substance_id)

    def get_by_cas(self, cas: str) -> Substance | None:
        """Get a substance by CAS number."""
        return self._by_cas.get(cas)

    def get_by_un(self, un: str) -> Substance | None:
        """Get a substance by UN number."""
        return self._by_un.get(un)

    def get_by_name(self, name: str) -> Substance | None:
        """Get a substance by exact name (case-insensitive)."""
        return self._by_name_lower.get(name.lower())

    def __contains__(self, substance_id: str) -> bool:
        return substance_id in self._substances

    def __getitem__(self, substance_id: str) -> Substance:
        sub = self._substances.get(substance_id)
        if sub is None:
            raise KeyError(f"Substance '{substance_id}' not found in database")
        return sub

    def __len__(self) -> int:
        return len(self._substances)

    def __iter__(self) -> Iterator[Substance]:
        return iter(self._substances.values())

    # ──────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20) -> List[Substance]:
        """Search substances by name, CAS, UN, or formula.

        Matches are case-insensitive and support partial/substring matching.
        Results are ranked: exact matches first, then prefix matches, then substring.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of matching Substance objects.
        """
        q = query.lower().strip()
        if not q:
            return list(self._substances.values())[:max_results]

        exact_matches: List[Substance] = []
        prefix_matches: List[Substance] = []
        substring_matches: List[Substance] = []
        seen: Set[str] = set()

        for sub in self._substances.values():
            if sub.id in seen:
                continue

            name_l = sub.name.lower()
            cas_l = (sub.cas_number or "").lower()
            un_l = (sub.un_number or "").lower()
            formula_l = (sub.formula or "").lower()

            # Rank 1: exact ID match
            if sub.id.lower() == q:
                if sub.id not in seen:
                    exact_matches.insert(0, sub)
                    seen.add(sub.id)
                continue

            # Rank 2: exact name match
            if name_l == q:
                if sub.id not in seen:
                    exact_matches.append(sub)
                    seen.add(sub.id)
                continue

            # Rank 3: exact CAS or UN match
            if cas_l == q or un_l == q:
                if sub.id not in seen:
                    exact_matches.append(sub)
                    seen.add(sub.id)
                continue

            # Rank 4: name prefix match
            if name_l.startswith(q):
                if sub.id not in seen:
                    prefix_matches.append(sub)
                    seen.add(sub.id)
                continue

            # Rank 5: substring in name, CAS, UN, or formula
            if (q in name_l or q in cas_l or q in un_l or
                    (formula_l and q in formula_l)):
                if sub.id not in seen:
                    substring_matches.append(sub)
                    seen.add(sub.id)

        results = exact_matches + prefix_matches + substring_matches
        return results[:max_results]

    def filter_by_hazard(self, hazard_class: str) -> List[Substance]:
        """Filter substances by hazard classification.

        Args:
            hazard_class: One of the HazardClass values ('flammable', 'toxic', etc.).

        Returns:
            List of matching Substance objects.
        """
        return [s for s in self._substances.values()
                if hazard_class in s.hazard_classes]

    def filter_by_phase(self, phase: str) -> List[Substance]:
        """Filter substances by ambient phase.

        Args:
            phase: 'gas' or 'liquid'.

        Returns:
            List of matching Substance objects.
        """
        return [s for s in self._substances.values()
                if s.phase_at_ambient == phase]

    def filter_by_tag(self, tag: str) -> List[Substance]:
        """Filter substances by tag."""
        return [s for s in self._substances.values() if tag in s.tags]

    def list_ids(self) -> List[str]:
        """Return all substance IDs, sorted."""
        return sorted(self._substances.keys())

    def list_names(self) -> List[str]:
        """Return all substance display names, sorted."""
        return sorted(s.name for s in self._substances.values())

    def list_tags(self) -> List[str]:
        """Return all unique tags in the database."""
        tags: Set[str] = set()
        for s in self._substances.values():
            tags.update(s.tags)
        return sorted(tags)

    # ──────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Total number of substances."""
        return len(self._substances)

    @property
    def count_flammable(self) -> int:
        return len(self.filter_by_hazard("flammable"))

    @property
    def count_toxic(self) -> int:
        return len(self.filter_by_hazard("toxic"))

    @property
    def count_gases(self) -> int:
        return len(self.filter_by_phase("gas"))

    @property
    def count_liquids(self) -> int:
        return len(self.filter_by_phase("liquid"))

    def stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        return {
            "version": self._version,
            "path": str(self._path),
            "total": self.count,
            "flammable": self.count_flammable,
            "toxic": self.count_toxic,
            "gases": self.count_gases,
            "liquids": self.count_liquids,
            "tags": self.list_tags(),
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        loaded = "✓" if self._loaded else "✗"
        return (f"SubstanceDatabase({self._path.name!r}, "
                f"v{self._version}, "
                f"{self.count} substances, loaded={loaded})")


# ══════════════════════════════════════════════════════════════════════════════
# Auto-load default database
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_DB: SubstanceDatabase | None = None


def get_database(db_path: str | Path | None = None) -> SubstanceDatabase:
    """Return the default substance database (singleton).

    Loads automatically on first call.

    Args:
        db_path: Optional path override.

    Returns:
        Loaded SubstanceDatabase instance.
    """
    global _DEFAULT_DB
    if _DEFAULT_DB is None or (db_path and _DEFAULT_DB.path != Path(db_path)):
        _DEFAULT_DB = SubstanceDatabase(db_path)
        _DEFAULT_DB.load()
    return _DEFAULT_DB
