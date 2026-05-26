"""
Rekarisk QRA — Risk Matrix (ISO 17776 / API Style).

5×5 risk matrix classification for qualitative risk presentation,
following ISO 17776 (Petroleum and natural gas industries — Offshore
production installations — Major accident hazard management) and
API RP 752/753 (Management of Hazards Associated with Location of
Process Plant Buildings).

The matrix combines:
  - Likelihood (frequency) categories: Rare → Frequent
  - Consequence (severity) categories: Negligible → Catastrophic
  - Risk levels: Low, Medium, High, Extreme

References:
  - ISO 17776:2016 — Major Accident Hazard Management
  - API RP 752 — Management of Hazards (Process Plant Buildings)
  - CCPS/AIChE — Guidelines for Risk Based Process Safety
  - UKOOA — Guidelines for QRA
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class LikelihoodLevel(str, Enum):
    """Likelihood (frequency) categories for 5×5 risk matrix.

    Frequency ranges in events per year.
    """
    RARE = "rare"               # < 1e-5/yr
    UNLIKELY = "unlikely"       # 1e-5 to < 1e-4/yr
    POSSIBLE = "possible"       # 1e-4 to < 1e-3/yr
    LIKELY = "likely"           # 1e-3 to < 1e-2/yr
    FREQUENT = "frequent"       # ≥ 1e-2/yr

    @property
    def index(self) -> int:
        """Column index (0 = Rare, 4 = Frequent)."""
        return {
            LikelihoodLevel.RARE: 0,
            LikelihoodLevel.UNLIKELY: 1,
            LikelihoodLevel.POSSIBLE: 2,
            LikelihoodLevel.LIKELY: 3,
            LikelihoodLevel.FREQUENT: 4,
        }[self]

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def frequency_range(self) -> str:
        ranges = {
            LikelihoodLevel.RARE: "< 1×10⁻⁵",
            LikelihoodLevel.UNLIKELY: "1×10⁻⁵ – 1×10⁻⁴",
            LikelihoodLevel.POSSIBLE: "1×10⁻⁴ – 1×10⁻³",
            LikelihoodLevel.LIKELY: "1×10⁻³ – 1×10⁻²",
            LikelihoodLevel.FREQUENT: "≥ 1×10⁻²",
        }
        return ranges[self]

    @property
    def hex_color(self) -> str:
        colors = {
            LikelihoodLevel.RARE: "#E8F5E9",
            LikelihoodLevel.UNLIKELY: "#C8E6C9",
            LikelihoodLevel.POSSIBLE: "#FFF9C4",
            LikelihoodLevel.LIKELY: "#FFE0B2",
            LikelihoodLevel.FREQUENT: "#FFCDD2",
        }
        return colors[self]


class ConsequenceLevel(str, Enum):
    """Consequence (severity) categories for 5×5 risk matrix."""
    NEGLIGIBLE = "negligible"       # No injuries, minor damage
    MINOR = "minor"                 # Minor injuries, limited damage
    MODERATE = "moderate"           # Serious injuries, local damage
    MAJOR = "major"                 # Single fatality, major damage
    CATASTROPHIC = "catastrophic"   # Multiple fatalities, extensive damage

    @property
    def index(self) -> int:
        """Row index (0 = Negligible, 4 = Catastrophic)."""
        return {
            ConsequenceLevel.NEGLIGIBLE: 0,
            ConsequenceLevel.MINOR: 1,
            ConsequenceLevel.MODERATE: 2,
            ConsequenceLevel.MAJOR: 3,
            ConsequenceLevel.CATASTROPHIC: 4,
        }[self]

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def fatality_range(self) -> str:
        ranges = {
            ConsequenceLevel.NEGLIGIBLE: "0",
            ConsequenceLevel.MINOR: "< 0.01 (injuries only)",
            ConsequenceLevel.MODERATE: "0.01 – 0.1",
            ConsequenceLevel.MAJOR: "0.1 – 1.0",
            ConsequenceLevel.CATASTROPHIC: "> 1.0",
        }
        return ranges[self]


class RiskLevel(str, Enum):
    """Risk level from the 5×5 matrix."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def hex_color(self) -> str:
        colors = {
            RiskLevel.LOW: "#4CAF50",      # Green
            RiskLevel.MEDIUM: "#FFEB3B",   # Yellow
            RiskLevel.HIGH: "#FF9800",     # Orange
            RiskLevel.EXTREME: "#F44336",  # Red
        }
        return colors[self]

    @property
    def rgb_color(self) -> tuple[int, int, int]:
        colors = {
            RiskLevel.LOW: (76, 175, 80),
            RiskLevel.MEDIUM: (255, 235, 59),
            RiskLevel.HIGH: (255, 152, 0),
            RiskLevel.EXTREME: (244, 67, 54),
        }
        return colors[self]


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RiskMatrixEntry:
    """A cell in the risk matrix."""
    likelihood: LikelihoodLevel
    consequence: ConsequenceLevel
    risk_level: RiskLevel
    description: str = ""

    @property
    def row(self) -> int:
        return self.consequence.index

    @property
    def col(self) -> int:
        return self.likelihood.index


# ──────────────────────────────────────────────────────────────────────
# Default 5×5 Risk Matrix (ISO 17776)
# ──────────────────────────────────────────────────────────────────────

# Matrix layout: rows = consequence (N→C, top→bottom), cols = likelihood (R→F, left→right)
# Risk Level:
#   L = Low, M = Medium, H = High, E = Extreme

DEFAULT_MATRIX: list[list[RiskLevel]] = [
    # Rare      Unlikely   Possible   Likely     Frequent
    [RiskLevel.LOW,    RiskLevel.LOW,    RiskLevel.LOW,    RiskLevel.MEDIUM,  RiskLevel.MEDIUM],   # Negligible
    [RiskLevel.LOW,    RiskLevel.LOW,    RiskLevel.MEDIUM, RiskLevel.MEDIUM,  RiskLevel.HIGH],     # Minor
    [RiskLevel.LOW,    RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH,    RiskLevel.HIGH],     # Moderate
    [RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH,   RiskLevel.HIGH,    RiskLevel.EXTREME], # Major
    [RiskLevel.MEDIUM, RiskLevel.HIGH,   RiskLevel.HIGH,   RiskLevel.EXTREME, RiskLevel.EXTREME], # Catastrophic
]

# Alternative: API RP 752 matrix (more conservative at upper end)
API_752_MATRIX: list[list[RiskLevel]] = [
    # Rare      Unlikely   Possible   Likely     Frequent
    [RiskLevel.LOW,    RiskLevel.LOW,    RiskLevel.MEDIUM, RiskLevel.MEDIUM,  RiskLevel.HIGH],     # Negligible
    [RiskLevel.LOW,    RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH,    RiskLevel.HIGH],     # Minor
    [RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH,   RiskLevel.HIGH,    RiskLevel.EXTREME], # Moderate
    [RiskLevel.MEDIUM, RiskLevel.HIGH,   RiskLevel.HIGH,   RiskLevel.EXTREME, RiskLevel.EXTREME], # Major
    [RiskLevel.HIGH,   RiskLevel.HIGH,   RiskLevel.EXTREME, RiskLevel.EXTREME, RiskLevel.EXTREME], # Catastrophic
]


# ──────────────────────────────────────────────────────────────────────
# Module Functions
# ──────────────────────────────────────────────────────────────────────

def classify_likelihood(frequency: float) -> LikelihoodLevel:
    """Classify a failure/event frequency into a likelihood category.

    Parameters
    ----------
    frequency : float
        Event frequency in events per year.

    Returns
    -------
    LikelihoodLevel
        Matrix likelihood category.

    Examples
    --------
    >>> classify_likelihood(5e-6)
    <LikelihoodLevel.RARE: 'rare'>
    >>> classify_likelihood(5e-2)
    <LikelihoodLevel.FREQUENT: 'frequent'>
    >>> classify_likelihood(5e-4)
    <LikelihoodLevel.POSSIBLE: 'possible'>
    """
    if frequency < 1e-5:
        return LikelihoodLevel.RARE
    elif frequency < 1e-4:
        return LikelihoodLevel.UNLIKELY
    elif frequency < 1e-3:
        return LikelihoodLevel.POSSIBLE
    elif frequency < 1e-2:
        return LikelihoodLevel.LIKELY
    else:
        return LikelihoodLevel.FREQUENT


def classify_consequence(
    fatalities: float = 0.0,
    injuries: float = 0.0,
) -> ConsequenceLevel:
    """Classify consequences into severity category based on fatalities.

    Uses the following thresholds:
    - Negligible: 0 fatalities, 0 injuries
    - Minor: < 0.01 fatalities (injuries only)
    - Moderate: 0.01 to < 0.1 fatalities
    - Major: 0.1 to < 1.0 fatalities
    - Catastrophic: ≥ 1.0 fatalities

    Parameters
    ----------
    fatalities : float
        Expected number of fatalities.
    injuries : float
        Expected number of injuries (used as tiebreaker for
        borderline cases).

    Returns
    -------
    ConsequenceLevel
        Severity category.

    Examples
    --------
    >>> classify_consequence(0.0).value
    'negligible'
    >>> classify_consequence(0.5).value
    'major'
    >>> classify_consequence(5.0).value
    'catastrophic'
    """
    if fatalities >= 1.0:
        return ConsequenceLevel.CATASTROPHIC
    elif fatalities >= 0.1:
        return ConsequenceLevel.MAJOR
    elif fatalities >= 0.01:
        return ConsequenceLevel.MODERATE
    elif fatalities > 0 or injuries > 0:
        return ConsequenceLevel.MINOR
    else:
        return ConsequenceLevel.NEGLIGIBLE


def classify_consequence_cost(
    cost_usd: float = 0.0,
) -> ConsequenceLevel:
    """Classify consequences based on estimated damage cost.

    Parameters
    ----------
    cost_usd : float
        Estimated damage cost in USD.

    Returns
    -------
    ConsequenceLevel
        Severity category.

    Thresholds:
    - Negligible: < $10,000
    - Minor: $10,000 – < $100,000
    - Moderate: $100,000 – < $1,000,000
    - Major: $1,000,000 – < $10,000,000
    - Catastrophic: ≥ $10,000,000
    """
    if cost_usd >= 10_000_000:
        return ConsequenceLevel.CATASTROPHIC
    elif cost_usd >= 1_000_000:
        return ConsequenceLevel.MAJOR
    elif cost_usd >= 100_000:
        return ConsequenceLevel.MODERATE
    elif cost_usd >= 10_000:
        return ConsequenceLevel.MINOR
    else:
        return ConsequenceLevel.NEGLIGIBLE


def risk_level(
    likelihood: Union[LikelihoodLevel, str],
    consequence: Union[ConsequenceLevel, str],
    matrix: Optional[list[list[RiskLevel]]] = None,
) -> RiskLevel:
    """Look up the risk level from the 5×5 matrix.

    Parameters
    ----------
    likelihood : LikelihoodLevel or str
        Likelihood (frequency) category.
    consequence : ConsequenceLevel or str
        Consequence (severity) category.
    matrix : list of list of RiskLevel, optional
        Custom matrix. Uses DEFAULT_MATRIX if None.

    Returns
    -------
    RiskLevel
        LOW, MEDIUM, HIGH, or EXTREME.

    Examples
    --------
    >>> risk_level("frequent", "catastrophic")
    <RiskLevel.EXTREME: 'extreme'>
    >>> risk_level("rare", "negligible")
    <RiskLevel.LOW: 'low'>
    """
    lh = LikelihoodLevel(likelihood) if isinstance(likelihood, str) else likelihood
    cs = ConsequenceLevel(consequence) if isinstance(consequence, str) else consequence
    mtx = matrix or DEFAULT_MATRIX

    row = cs.index
    col = lh.index

    return mtx[row][col]


def risk_level_from_values(
    frequency: float,
    fatalities: float = 0.0,
    cost_usd: float = 0.0,
) -> RiskLevel:
    """Convenience function: classify frequency and consequence, then look up risk.

    Parameters
    ----------
    frequency : float
        Event frequency (per year).
    fatalities : float
        Expected fatalities.
    cost_usd : float
        Estimated damage cost (USD).

    Returns
    -------
    RiskLevel
    """
    lh = classify_likelihood(frequency)
    cs = classify_consequence(fatalities)
    return risk_level(lh, cs)


def risk_matrix_table(
    matrix: Optional[list[list[RiskLevel]]] = None,
) -> list[list[dict[str, str]]]:
    """Generate the full 5×5 risk matrix as a table of dicts.

    Returns
    -------
    list of list of dict
        Each dict contains: likelihood, consequence, risk_level, color.

    Examples
    --------
    >>> table = risk_matrix_table()
    >>> len(table)  # 5 rows
    5
    >>> len(table[0])  # 5 columns
    5
    """
    mtx = matrix or DEFAULT_MATRIX
    cons_levels = list(ConsequenceLevel)
    like_levels = list(LikelihoodLevel)

    result: list[list[dict[str, str]]] = []
    for i, cs in enumerate(cons_levels):
        row: list[dict[str, str]] = []
        for j, lh in enumerate(like_levels):
            rl = mtx[i][j]
            row.append({
                "likelihood": lh.value,
                "consequence": cs.value,
                "risk_level": rl.value,
                "color": rl.hex_color,
                "description": f"{lh.label} × {cs.label} = {rl.label}",
            })
        result.append(row)
    return result


def risk_matrix_html(
    matrix: Optional[list[list[RiskLevel]]] = None,
    include_legend: bool = True,
) -> str:
    """Generate HTML representation of the 5×5 risk matrix.

    Useful for embedding in reports.

    Parameters
    ----------
    matrix : optional
        Custom risk matrix. Uses DEFAULT_MATRIX if None.
    include_legend : bool
        If True, include a legend below the matrix.

    Returns
    -------
    str
        HTML string.
    """
    mtx = matrix or DEFAULT_MATRIX
    cons_levels = list(ConsequenceLevel)
    like_levels = list(LikelihoodLevel)

    lines = [
        '<table style="border-collapse: collapse; text-align: center; font-family: sans-serif;">',
    ]

    # Header row
    lines.append('<tr><th style="padding: 8px; background: #ddd;">Consequence ↓ / Likelihood →</th>')
    for lh in like_levels:
        lines.append(
            f'<th style="padding: 8px; background: {lh.hex_color}; border: 1px solid #999;">'
            f'{lh.label}<br><small>{lh.frequency_range}</small></th>'
        )
    lines.append('</tr>')

    # Matrix rows
    for i, cs in enumerate(cons_levels):
        lines.append('<tr>')
        lines.append(
            f'<th style="padding: 8px; background: #ddd; border: 1px solid #999;">'
            f'{cs.label}<br><small>{cs.fatality_range}</small></th>'
        )
        for j, _ in enumerate(like_levels):
            rl = mtx[i][j]
            lines.append(
                f'<td style="padding: 12px; background: {rl.hex_color}; '
                f'border: 1px solid #999; font-weight: bold; color: '
                f'{"#fff" if rl in (RiskLevel.HIGH, RiskLevel.EXTREME) else "#333"};">'
                f'{rl.label}</td>'
            )
        lines.append('</tr>')

    lines.append('</table>')

    if include_legend:
        lines.append('<br><div style="font-family: sans-serif; font-size: 14px;">')
        lines.append('<b>Legend:</b><br>')
        for rl in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME):
            lines.append(
                f'<span style="display: inline-block; width: 16px; height: 16px; '
                f'background: {rl.hex_color}; border: 1px solid #999; margin-right: 8px;">'
                f'&nbsp;</span> {rl.label}<br>'
            )
        lines.append('</div>')

    return '\n'.join(lines)


def risk_matrix_json(indent: int = 2) -> str:
    """Export the risk matrix as JSON."""
    return json.dumps(risk_matrix_table(), indent=indent)


def get_matrix_summary() -> str:
    """Return a text summary of the risk matrix."""
    lines = ["Risk Matrix Summary (ISO 17776)", "=" * 40, ""]
    cons_levels = list(ConsequenceLevel)
    like_levels = list(LikelihoodLevel)

    for cs in cons_levels:
        for lh in like_levels:
            rl = risk_level(lh, cs)
            lines.append(
                f"  {lh.label:>10} × {cs.label:<14} → {rl.label}"
            )

    return '\n'.join(lines)
