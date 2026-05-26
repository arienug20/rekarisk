"""
Rekarisk Advanced Analysis — Worst-Case Scenario Identification.

Identifies worst-case scenarios from batch results and systematically
varies key parameters to find the combination that produces the most
extreme (hazardous) outcome.

Typical usage:
    batch_result = batch_runner.run(batch_input)
    worst_id, worst_val = find_worst_case(batch_result, "max_concentration")
    params = worst_case_parameters(batch_result, "max_concentration")
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Primary API: find worst scenario from batch
# ---------------------------------------------------------------------------


def find_worst_case(
    batch_result: Any,
    metric: str,
    direction: str = "maximize",
    n_worst: int = 1,
) -> List[Tuple[str, float]]:
    """Find the worst-case scenario(s) from batch results.

    Scans the summary table for the scenario with the most extreme value
    on the specified metric.

    Args:
        batch_result: BatchResult with summary_table from batch run.
        metric: The output metric to evaluate (e.g., 'max_concentration',
            'overpressure', 'heat_flux', 'impact_distance').
        direction: 'maximize' (higher = worse) or 'minimize' (lower = worse).
        n_worst: Number of worst scenarios to return (default 1).

    Returns:
        List of (scenario_id, metric_value) tuples, sorted worst-first.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class FakeBatch:
        ...     summary_table = [
        ...         {"scenario_id": "s0", "max_concentration": 100.0},
        ...         {"scenario_id": "s1", "max_concentration": 500.0},
        ...         {"scenario_id": "s2", "error": "failed"},
        ...     ]
        >>> results = find_worst_case(FakeBatch(), "max_concentration")
        >>> results[0][0]
        's1'
        >>> results[0][1]
        500.0
    """
    # Extract valid rows (no errors)
    valid_rows: List[Dict[str, Any]] = []
    for row in batch_result.summary_table:
        if "error" not in row and metric in row:
            val = row[metric]
            if isinstance(val, (int, float, np.floating)) and not np.isnan(float(val)):
                valid_rows.append(row)

    if not valid_rows:
        return []

    # Sort by metric
    reverse = (direction == "maximize")
    valid_rows.sort(
        key=lambda r: float(r[metric]),
        reverse=reverse,
    )

    worst: List[Tuple[str, float]] = []
    for row in valid_rows[:n_worst]:
        worst.append((str(row["scenario_id"]), float(row[metric])))

    return worst


# ---------------------------------------------------------------------------
# Analyze worst-case parameters
# ---------------------------------------------------------------------------


def worst_case_parameters(
    batch_result: Any,
    metric: str,
    direction: str = "maximize",
) -> Dict[str, Any]:
    """Extract the parameters that produce the worst-case scenario.

    From the batch results, identifies which input parameter combinations
    lead to the worst outcomes, providing guidance on what conditions
    are most hazardous.

    Args:
        batch_result: BatchResult from batch run.
        metric: Which output metric to evaluate.
        direction: 'maximize' (higher = worse) or 'minimize' (lower = worse).

    Returns:
        Dict with:
            - worst_scenario_id: ID of the worst scenario
            - worst_value: The extreme metric value
            - parameters: Dict of the input parameters for the worst case
            - parameter_deviations: For numeric params, deviation from
              average across all scenarios

    Examples:
        >>> from dataclasses import dataclass, field
        >>> from typing import List, Tuple
        >>> @dataclass
        ... class FakeBatch:
        ...     summary_table: List[Dict] = field(default_factory=list)
        ...     results: List[Tuple] = field(default_factory=list)
        >>> fb = FakeBatch(
        ...     summary_table=[
        ...         {"scenario_id": "s0", "max_concentration": 100.0},
        ...         {"scenario_id": "s1", "max_concentration": 500.0},
        ...     ],
        ...     results=[
        ...         ("s0", {"source_rate": 1.0, "wind_speed": 5.0}, {}),
        ...         ("s1", {"source_rate": 10.0, "wind_speed": 1.0}, {}),
        ...     ],
        ... )
        >>> params = worst_case_parameters(fb, "max_concentration")
        >>> params["worst_scenario_id"]
        's1'
        >>> params["parameters"]["source_rate"]
        10.0
    """
    worst_scenarios = find_worst_case(batch_result, metric, direction, n_worst=1)
    if not worst_scenarios:
        return {
            "worst_scenario_id": None,
            "worst_value": None,
            "parameters": {},
            "parameter_deviations": {},
        }

    worst_id, worst_val = worst_scenarios[0]

    # Find the scenario params from results
    worst_params: Dict[str, Any] = {}
    for scenario_id, scenario, _ in batch_result.results:
        if scenario_id == worst_id:
            worst_params = dict(scenario)
            break

    # Collect all parameters and compute averages for numeric ones
    all_numeric_params: Dict[str, List[float]] = defaultdict(list)
    for _, scenario, _ in batch_result.results:
        for key, value in scenario.items():
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
                all_numeric_params[key].append(float(value))

    # Compute deviations
    deviations: Dict[str, float] = {}
    for key, wval in worst_params.items():
        if not isinstance(wval, (int, float, np.floating)) or isinstance(wval, bool):
            continue
        vals = all_numeric_params.get(key, [])
        if len(vals) > 1:
            avg = np.mean(vals)
            if abs(avg) > 1e-12:
                deviations[key] = (float(wval) - float(avg)) / abs(avg)
            else:
                deviations[key] = 0.0
        else:
            deviations[key] = 0.0

    return {
        "worst_scenario_id": worst_id,
        "worst_value": worst_val,
        "parameters": worst_params,
        "parameter_deviations": deviations,
    }


# ---------------------------------------------------------------------------
# Systematic worst-case search
# ---------------------------------------------------------------------------


def systematic_worst_case_search(
    param_ranges: Dict[str, Tuple[float, float]],
    model_function: Callable,
    output_extractor: Callable[[Any], float],
    n_steps: int = 5,
    param_fixed: Optional[Dict[str, float]] = None,
    direction: str = "maximize",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Systematically search parameter space for worst-case combination.

    Evaluates the model on a grid of parameter values within the specified
    ranges and finds the combination that produces the worst outcome.

    Args:
        param_ranges: Dict of {param_name: (min, max)} defining the search space.
        model_function: Callable(**params) → result.
        output_extractor: Callable(result) → float (e.g., lambda r: r.max_concentration).
        n_steps: Number of grid steps per parameter dimension.
        param_fixed: Fixed parameter values not being varied.
        direction: 'maximize' or 'minimize'.
        verbose: Print search progress.

    Returns:
        Dict with worst_params, worst_value, search_points, elapsed_time.

    Examples:
        >>> def model(a=1.0, b=2.0):
        ...     return a * b
        >>> result = systematic_worst_case_search(
        ...     param_ranges={"a": (1.0, 10.0), "b": (1.0, 5.0)},
        ...     model_function=model,
        ...     output_extractor=lambda r: r,
        ...     n_steps=3,
        ...     direction="maximize",
        ... )
        >>> result["worst_params"]["a"]
        10.0
        >>> result["worst_params"]["b"]
        5.0
    """
    import itertools
    import time

    start_time = time.monotonic()

    param_names = list(param_ranges.keys())
    param_grids = {}
    for name in param_names:
        lo, hi = param_ranges[name]
        param_grids[name] = np.linspace(lo, hi, n_steps)

    base = deepcopy(param_fixed) if param_fixed else {}

    worst_value = float("-inf") if direction == "maximize" else float("inf")
    worst_params: Dict[str, float] = {}
    search_points: List[Dict[str, Any]] = []

    total_combos = n_steps ** len(param_names)
    count = 0

    for combo in itertools.product(*param_grids.values()):
        count += 1
        kwargs = dict(base)
        for i, name in enumerate(param_names):
            kwargs[name] = float(combo[i])

        try:
            result = model_function(**kwargs)
            val = output_extractor(result)
        except Exception:
            continue

        search_points.append({
            "params": dict(kwargs),
            "output": float(val),
        })

        is_worse = (
            (direction == "maximize" and val > worst_value)
            or (direction == "minimize" and val < worst_value)
        )
        if is_worse:
            worst_value = val
            worst_params = dict(kwargs)

        if verbose and count % max(1, total_combos // 10) == 0:
            print(f"[WorstCase] {count}/{total_combos} evaluated...")

    elapsed = time.monotonic() - start_time

    if verbose:
        print(
            f"[WorstCase] Done. {count} points evaluated in {elapsed:.2f}s. "
            f"Worst value: {worst_value:.4g}"
        )

    return {
        "worst_params": worst_params,
        "worst_value": worst_value if worst_params else None,
        "direction": direction,
        "n_evaluated": count,
        "search_points": search_points,
        "elapsed_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def classify_severity(
    value: float,
    thresholds: Dict[str, float],
    ascending: bool = True,
) -> str:
    """Classify a consequence value into a severity category.

    Args:
        value: The consequence metric value.
        thresholds: Dict mapping category name → threshold value.
            Categories are checked in order.
        ascending: If True, categories apply when value < threshold
            (e.g., lower is worse for dose). If False, applies when
            value > threshold (e.g., higher is worse for concentration).

    Returns:
        Category name, or 'unknown'.
    """
    for category, threshold in thresholds.items():
        if ascending:
            if value < threshold:
                return category
        else:
            if value > threshold:
                return category
    return list(thresholds.keys())[-1] if thresholds else "unknown"
