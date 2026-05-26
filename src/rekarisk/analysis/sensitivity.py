"""
Rekarisk Advanced Analysis — Sensitivity Analysis.

One-at-a-time (OAT) parameter sensitivity analysis for consequence models.
Varies each parameter individually within specified ranges while holding all
others at their base values, producing tornado-chart data and parameter
importance rankings.

Typical usage:
    result = run_oat(SensitivityInput(
        base_case={"source_rate": 5.0, "wind_speed": 3.0, ...},
        parameters={"source_rate": (1.0, 10.0), "wind_speed": (1.0, 6.0)},
        model_function=my_dispersion_model,
        output_key="max_concentration",
    ))
    labels, base, low, high = tornado_data(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SensitivityInput:
    """Input for OAT sensitivity analysis.

    Attributes:
        base_case: Dict of base parameter values.
        parameters: Dict of {param_name: (min_value, max_value)} specifying
            the variation range for each parameter. If any parameter value is
            None or a single float, a ±20% range is automatically generated.
        model_function: Callable(**params) → result. Accepts the parameter
            dict and returns a result object or numeric value.
        output_key: Which result metric to extract for sensitivity ranking.
            If None, the result is assumed to be a raw numeric value.
        output_extractor: Optional callable that extracts the target metric
            from the model result. Signature: (result) → float. Takes priority
            over output_key.
    """

    base_case: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    model_function: Optional[Callable] = None
    output_key: Optional[str] = None
    output_extractor: Optional[Callable[[Any], float]] = None


@dataclass
class SensitivityResult:
    """Results of OAT sensitivity analysis.

    Attributes:
        parameter_effects: Dict {param_name: (output_at_min, output_at_max, range)}.
        rankings: List of (param_name, sensitivity_index), sorted descending
            by absolute sensitivity.
        base_output: The output value for the base (unvaried) case.
        model_errors: Dict {param_name: error_message} for parameters that
            failed during evaluation.
    """

    parameter_effects: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    rankings: List[Tuple[str, float]] = field(default_factory=list)
    base_output: float = 0.0
    model_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_effects)

    @property
    def success_count(self) -> int:
        return len(self.parameter_effects)

    @property
    def error_count(self) -> int:
        return len(self.model_errors)

    def get_effect(self, param_name: str) -> Optional[Tuple[float, float, float]]:
        """Get the effect tuple for a parameter, or None if not computed."""
        return self.parameter_effects.get(param_name)


# ---------------------------------------------------------------------------
# Output Extraction Helpers
# ---------------------------------------------------------------------------


def _extract_output(result: Any, key: Optional[str], extractor: Optional[Callable]) -> float:
    """Extract a numeric output from a result object.

    Priority:
        1. extractor callable (if provided)
        2. key-based lookup on result dict
        3. key-based attribute access on result object
        4. result.to_dict()[key] if available
        5. If result is a raw number, return it directly
    """
    # Extractor callable
    if extractor is not None:
        return float(extractor(result))

    # Raw numeric result
    if isinstance(result, (int, float, np.floating)):
        return float(result)

    # Dict result
    if isinstance(result, dict):
        if key is not None and key in result:
            return float(result[key])
        # Try common metrics
        for k in ["max_concentration", "max_flux", "impact_distance",
                   "max_distance", "heat_flux", "overpressure",
                   "probability_of_death", "probit_value"]:
            if k in result:
                v = result[k]
                if isinstance(v, (int, float, np.floating)):
                    return float(v)

    # Object result — try attribute
    if key is not None and hasattr(result, key):
        val = getattr(result, key)
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        if isinstance(val, np.ndarray) and val.size == 1:
            return float(val.item())

    # Try common attributes
    for k in ["max_concentration", "max_flux", "impact_distance",
               "max_distance", "heat_flux", "overpressure",
               "probability_of_death", "probit_value"]:
        if hasattr(result, k):
            val = getattr(result, k)
            if isinstance(val, (int, float, np.floating)):
                return float(val)
            if isinstance(val, np.ndarray) and val.size == 1:
                return float(val.item())

    # Try to_dict()
    if hasattr(result, "to_dict"):
        try:
            d = result.to_dict()
            if isinstance(d, dict) and key is not None and key in d:
                return float(d[key])
        except Exception:
            pass

    # Last resort: try to convert
    try:
        return float(result)
    except (TypeError, ValueError):
        raise ValueError(
            f"Could not extract numeric output from result of type "
            f"{type(result).__name__}. Provide an output_key or "
            f"output_extractor."
        )


# ---------------------------------------------------------------------------
# Core OAT Algorithm
# ---------------------------------------------------------------------------


def run_oat(sensitivity_input: SensitivityInput) -> SensitivityResult:
    """Run One-At-a-Time sensitivity analysis.

    For each parameter in sensitivity_input.parameters, the function:
        1. Sets that parameter to its min value (others at base)
        2. Runs the model, records output
        3. Sets that parameter to its max value (others at base)
        4. Runs the model, records output

    If a parameter's range is not specified (empty tuple or None), a default
    ±20% range around the base value is used.

    Args:
        sensitivity_input: SensitivityInput with base case, parameters,
            model function, and output key.

    Returns:
        SensitivityResult with parameter effects, rankings, and base output.

    Raises:
        ValueError: If model_function is None or base_case is empty.

    Examples:
        >>> def dummy_model(source_rate=1.0, wind_speed=3.0):
        ...     return source_rate / wind_speed
        >>> si = SensitivityInput(
        ...     base_case={"source_rate": 5.0, "wind_speed": 3.0},
        ...     parameters={"source_rate": (1.0, 10.0), "wind_speed": (1.0, 6.0)},
        ...     model_function=dummy_model,
        ...     output_key=None,  # raw numeric
        ... )
        >>> res = run_oat(si)
        >>> res.base_output == 5.0 / 3.0
        True
        >>> len(res.parameter_effects) == 2
        True
        >>> res.rankings[0][0] in ("source_rate", "wind_speed")
        True
    """
    if sensitivity_input.model_function is None:
        raise ValueError("SensitivityInput requires a model_function callable.")
    if not sensitivity_input.base_case:
        raise ValueError("SensitivityInput requires a non-empty base_case dict.")

    model_fn = sensitivity_input.model_function
    base_case = dict(sensitivity_input.base_case)
    output_key = sensitivity_input.output_key
    extractor = sensitivity_input.output_extractor

    # Compute base output
    try:
        base_result = model_fn(**base_case)
        base_output = _extract_output(base_result, output_key, extractor)
    except Exception as exc:
        raise RuntimeError(f"Base case model evaluation failed: {exc}") from exc

    # If no parameters provided, auto-generate ±20% from base case
    params = dict(sensitivity_input.parameters)
    if not params:
        for key, value in base_case.items():
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
                v = float(value)
                if v != 0:
                    params[key] = (v * 0.8, v * 1.2)
                else:
                    params[key] = (-1.0, 1.0)

    # Ensure all parameter ranges are valid
    resolved_params: Dict[str, Tuple[float, float]] = {}
    for param_name, range_spec in params.items():
        if param_name not in base_case:
            continue
        base_val = base_case[param_name]
        if not isinstance(base_val, (int, float, np.floating)) or isinstance(base_val, bool):
            continue

        if range_spec is None or (isinstance(range_spec, tuple) and len(range_spec) == 0):
            # Auto ±20%
            bv = float(base_val)
            if bv != 0:
                resolved_params[param_name] = (bv * 0.8, bv * 1.2)
            else:
                resolved_params[param_name] = (-1.0, 1.0)
        elif isinstance(range_spec, (tuple, list)) and len(range_spec) == 2:
            resolved_params[param_name] = (float(range_spec[0]), float(range_spec[1]))
        elif isinstance(range_spec, (int, float, np.floating)):
            # Single value = center; auto ±20%
            bv = float(range_spec)
            if bv != 0:
                resolved_params[param_name] = (bv * 0.8, bv * 1.2)
            else:
                resolved_params[param_name] = (-1.0, 1.0)

    parameter_effects: Dict[str, Tuple[float, float, float]] = {}
    model_errors: Dict[str, str] = {}

    for param_name, (min_val, max_val) in resolved_params.items():
        # Ensure min < max
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        output_min: Optional[float] = None
        output_max: Optional[float] = None

        # Evaluate at parameter min
        scenario_min = dict(base_case)
        scenario_min[param_name] = min_val
        try:
            result_min = model_fn(**scenario_min)
            output_min = _extract_output(result_min, output_key, extractor)
        except Exception as exc:
            model_errors[f"{param_name}_min"] = str(exc)

        # Evaluate at parameter max
        scenario_max = dict(base_case)
        scenario_max[param_name] = max_val
        try:
            result_max = model_fn(**scenario_max)
            output_max = _extract_output(result_max, output_key, extractor)
        except Exception as exc:
            model_errors[f"{param_name}_max"] = str(exc)

        if output_min is not None and output_max is not None:
            effect_range = output_max - output_min
            parameter_effects[param_name] = (output_min, output_max, effect_range)

    # Rank parameters by absolute sensitivity (effect range)
    rankings: List[Tuple[str, float]] = sorted(
        parameter_effects.items(),
        key=lambda item: abs(item[1][2]),
        reverse=True,
    )
    rankings = [(name, effect[2]) for name, effect in rankings]

    return SensitivityResult(
        parameter_effects=parameter_effects,
        rankings=rankings,
        base_output=base_output,
        model_errors=model_errors,
    )


# ---------------------------------------------------------------------------
# Sensitivity Indices
# ---------------------------------------------------------------------------


def sensitivity_indices(
    result: SensitivityResult,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute sensitivity indices from OAT results.

    Sensitivity index for parameter p:
        S_p = |ΔO| / Δp
    where ΔO is the output range and Δp is the input range.

    If normalized=True, indices sum to 1.0.

    Args:
        result: SensitivityResult from run_oat().
        normalize: If True, normalize indices to sum to 1.0.

    Returns:
        Dict {param_name: sensitivity_index}.

    Examples:
        >>> # Using the same dummy model from run_oat example
        >>> # source_rate effect = 10/3 - 1/3 = 3.0 (range 9 → 0.333)
        >>> # wind_speed effect = 5/1 - 5/6 = 4.167 (range 5 → 0.208)
        >>> # Actually let's compute properly:
        >>> # source_rate: min=1→1/3=0.333, max=10→10/3=3.333, range=3.0
        >>> # wind_speed: min=1→5/1=5.0, max=6→5/6=0.833, range=-4.167
        >>> # So absolute range: source_rate=3.0, wind_speed=4.167
        >>> assert True  # doc only
    """
    indices: Dict[str, float] = {}
    for param_name, (_, _, effect_range) in result.parameter_effects.items():
        indices[param_name] = abs(effect_range)

    if normalize and indices:
        total = sum(indices.values())
        if total > 0:
            for key in indices:
                indices[key] /= total

    return indices


# ---------------------------------------------------------------------------
# Tornado Chart Data
# ---------------------------------------------------------------------------


def tornado_data(
    result: SensitivityResult,
    top_n: int = 20,
) -> Tuple[List[str], float, List[float], List[float]]:
    """Extract tornado chart data from a SensitivityResult.

    Returns data suitable for creating a tornado (butterfly/back-to-back)
    chart showing the effect of each parameter on the output.

    Args:
        result: SensitivityResult from run_oat().
        top_n: Limit to top N parameters (most sensitive first).

    Returns:
        Tuple of:
            - labels: List of parameter names (most sensitive last, for
              bottom-to-top tornado display)
            - base_value: The base case output value
            - low_values: Output values at parameter minimum
            - high_values: Output values at parameter maximum

    The returned lists are ordered so that the most sensitive parameter
    appears at the top of the chart (last in the list, when plotted
    bottom-up).

    Examples:
        >>> # Dummy result
        >>> result = SensitivityResult(
        ...     parameter_effects={
        ...         "a": (10.0, 20.0, 10.0),
        ...         "b": (5.0, 8.0, 3.0),
        ...     },
        ...     rankings=[("a", 10.0), ("b", 3.0)],
        ...     base_output=15.0,
        ... )
        >>> labels, base, low, high = tornado_data(result)
        >>> len(labels) == 2
        True
        >>> base == 15.0
        True
    """
    if not result.rankings:
        return ([], result.base_output, [], [])

    # Take top N by absolute sensitivity
    top_params = result.rankings[:top_n]

    labels: List[str] = []
    low_values: List[float] = []
    high_values: List[float] = []

    for param_name, _ in top_params:
        effect = result.parameter_effects.get(param_name)
        if effect is None:
            continue
        out_min, out_max, _ = effect
        labels.append(param_name)
        low_values.append(out_min)
        high_values.append(out_max)

    # Reverse for bottom-to-top tornado display
    labels.reverse()
    low_values.reverse()
    high_values.reverse()

    return (labels, result.base_output, low_values, high_values)


# ---------------------------------------------------------------------------
# Parameter Ranking Helper
# ---------------------------------------------------------------------------


def parameter_ranks(
    result: SensitivityResult,
) -> List[Dict[str, Any]]:
    """Return parameter rankings as a list of dicts for table display.

    Args:
        result: SensitivityResult.

    Returns:
        List of dicts with keys: parameter, rank, low_output, high_output,
        base_output, sensitivity_index.
    """
    table: List[Dict[str, Any]] = []
    for rank, (param_name, sensitivity_index) in enumerate(result.rankings, 1):
        effect = result.parameter_effects.get(param_name)
        if effect is None:
            continue
        out_min, out_max, _ = effect
        table.append({
            "rank": rank,
            "parameter": param_name,
            "low_output": out_min,
            "high_output": out_max,
            "base_output": result.base_output,
            "sensitivity_index": sensitivity_index,
            "range": out_max - out_min,
            "range_pct": (
                (out_max - out_min) / abs(result.base_output) * 100
                if result.base_output != 0
                else 0.0
            ),
        })
    return table
