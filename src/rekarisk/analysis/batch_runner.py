"""
Rekarisk Advanced Analysis — Batch Runner.

Runs multiple consequence scenarios in sequence or parallel, collecting
results, generating summary tables, computing aggregate statistics,
and tracking individual failures without halting the entire batch.

Usage:
    runner = BatchRunner()
    batch_input = BatchInput(
        scenarios=[scenario1, scenario2],
        weather_set=[weather1, weather2],
    )
    result = runner.run(batch_input, callback=my_callback)
    print(result.summary_table)
"""

from __future__ import annotations

import gc
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class BatchInput:
    """Input specification for batch run.

    Attributes:
        scenarios: List of scenario dicts. Each scenario is a complete model
            input dict (e.g., PlumeInput fields or ReleaseInfo fields).
        weather_set: List of weather condition dicts (multi-weather).
            If empty, scenarios are run standalone with their own weather.
        substance_list: List of substance identifiers or dicts (multi-substance).
            If empty, scenarios use their own substance.
        combinations: If True, run all weather × substance × scenario
            combinations. Default False — only 1:1:1 mapping.
        model_function: Callable that takes a scenario dict and returns a
            result object. Must accept keyword args corresponding to dict keys.
        parallel: Number of parallel workers (0 = sequential, >0 = multiprocessing).
            Default 0.
        verbose: Print progress to stdout. Default False.
    """

    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    weather_set: List[Dict[str, Any]] = field(default_factory=list)
    substance_list: List[Any] = field(default_factory=list)
    combinations: bool = False
    model_function: Optional[Callable] = None
    parallel: int = 0
    verbose: bool = False


@dataclass
class BatchResult:
    """Aggregated results from a batch run.

    Attributes:
        results: List of (scenario_id, scenario, result) tuples.
        summary_table: List of dicts with key outputs per scenario.
        statistics: Dict of {metric: {min, max, mean, std}} across scenarios.
        failed_scenarios: List of (scenario_id, error_message) tuples.
        total_scenarios: Total number of scenarios attempted.
        elapsed_time: Wall-clock time for the batch run [s].
    """

    results: List[Tuple[str, Dict[str, Any], Any]] = field(default_factory=list)
    summary_table: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failed_scenarios: List[Tuple[str, str]] = field(default_factory=list)
    total_scenarios: int = 0
    elapsed_time: float = 0.0

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def failure_count(self) -> int:
        return len(self.failed_scenarios)


# ---------------------------------------------------------------------------
# Helper — extract key metrics from a result object
# ---------------------------------------------------------------------------


_METRIC_KEYS = [
    "max_concentration",
    "max_distance",
    "max_flux",
    "impact_distance",
    "plume_rise_delta",
    "transition_distance",
    "pool_radius",
    "flame_length",
    "flame_height",
    "surface_emissive_power",
    "heat_flux",
    "overpressure",
    "impulse",
    "probit_value",
    "probability_of_death",
    "risk_individual",
    "source_mass",
    "source_rate",
]


def _extract_metrics(result: Any) -> Dict[str, float]:
    """Extract numeric metrics from a result object or dict."""
    metrics: Dict[str, float] = {}

    if isinstance(result, dict):
        for key in _METRIC_KEYS:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float, np.floating)):
                    metrics[key] = float(val)
        # Also capture any additional numeric top-level keys
        for key, value in result.items():
            if key not in metrics and isinstance(value, (int, float, np.floating)):
                metrics[key] = float(value)
        return metrics

    # Object with attributes
    for key in _METRIC_KEYS:
        if hasattr(result, key):
            val = getattr(result, key)
            if isinstance(val, (int, float, np.floating, np.ndarray)):
                if isinstance(val, np.ndarray) and val.size == 1:
                    metrics[key] = float(val.item())
                elif isinstance(val, (int, float, np.floating)):
                    metrics[key] = float(val)

    # Try to_dict if available
    if hasattr(result, "to_dict"):
        try:
            d = result.to_dict()
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float, bool, np.floating)) and k not in metrics:
                        metrics[k] = float(v)
        except Exception:
            pass

    return metrics


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge b into a."""
    merged = dict(a)
    merged.update(b)
    return merged


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------


class BatchRunner:
    """Execute multiple consequence scenarios and aggregate results.

    Supports sequential and multiprocessing-based parallel execution with
    per-scenario error handling, progress tracking, and memory management.

    Usage:
        runner = BatchRunner()
        result = runner.run(batch_input, callback=progress_fn)
    """

    def __init__(self):
        self._start_time: float = 0.0
        self._gc_counter: int = 0
        self._gc_frequency: int = 10  # clear memory every N scenarios

    def run(
        self,
        batch_input: BatchInput,
        callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """Run the batch of scenarios.

        Args:
            batch_input: BatchInput with scenarios, weather, substances, etc.
            callback: Optional progress callback(n_done, n_total, status_msg).

        Returns:
            BatchResult with all scenario results and aggregate statistics.

        Raises:
            ValueError: If no model_function is provided.
        """
        if batch_input.model_function is None:
            raise ValueError("BatchInput requires a model_function callable.")

        self._start_time = time.monotonic()
        self._gc_counter = 0

        # Build the scenario queue
        queue = self._build_queue(batch_input)
        total = len(queue)

        if total == 0:
            return BatchResult(total_scenarios=0,
                               elapsed_time=time.monotonic() - self._start_time)

        if batch_input.verbose:
            print(f"[BatchRunner] {total} scenario(s) queued.")

        results: List[Tuple[str, Dict[str, Any], Any]] = []
        failures: List[Tuple[str, str]] = []
        summary_rows: List[Dict[str, Any]] = []

        if batch_input.parallel > 0:
            # Multiprocessing path
            results, failures, summary_rows = self._run_parallel(
                queue, batch_input, callback
            )
        else:
            # Sequential path
            for idx, (scenario_id, scenario) in enumerate(queue):
                if batch_input.verbose:
                    print(f"[BatchRunner] [{idx + 1}/{total}] {scenario_id}")

                if callback:
                    callback(idx + 1, total, f"Running: {scenario_id}")

                result = None  # ensure binding exists for cleanup
                try:
                    result = batch_input.model_function(**scenario)
                    results.append((scenario_id, scenario, result))

                    # Extract summary metrics
                    metrics = _extract_metrics(result)
                    row = {"scenario_id": scenario_id, **metrics}
                    summary_rows.append(row)

                    if callback:
                        callback(idx + 1, total, f"OK: {scenario_id}")
                except Exception as exc:
                    tb = traceback.format_exc()
                    err_msg = f"{type(exc).__name__}: {exc}"
                    failures.append((scenario_id, err_msg))
                    # Record failure in summary too
                    row = {"scenario_id": scenario_id, "error": err_msg}
                    summary_rows.append(row)

                    if batch_input.verbose:
                        print(f"[BatchRunner] FAILED: {scenario_id} — {err_msg}")
                        print(tb)
                    if callback:
                        callback(idx + 1, total, f"FAILED: {scenario_id} — {err_msg}")

                # Memory management: periodically clear large arrays
                self._gc_counter += 1
                if self._gc_counter >= self._gc_frequency:
                    gc.collect()
                    self._gc_counter = 0

                # Clear intermediate results if available
                if result is not None:
                    try:
                        if hasattr(result, "concentration_grid"):
                            del result.concentration_grid
                        if hasattr(result, "ground_concentration"):
                            del result.ground_concentration
                    except Exception:
                        pass

        # Compute aggregate statistics
        statistics = self._compute_statistics(summary_rows)

        elapsed = time.monotonic() - self._start_time

        if batch_input.verbose:
            print(
                f"[BatchRunner] Done. {len(results)}/{total} succeeded, "
                f"{len(failures)} failed in {elapsed:.2f}s."
            )

        return BatchResult(
            results=results,
            summary_table=summary_rows,
            statistics=statistics,
            failed_scenarios=failures,
            total_scenarios=total,
            elapsed_time=elapsed,
        )

    def _build_queue(
        self, batch_input: BatchInput
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Build the scenario queue from BatchInput.

        If combinations=True, creates all weather × substance × scenario combos.
        Otherwise, builds a 1:1 queue with any weather/substance merged.
        """
        scenarios = batch_input.scenarios
        weathers = batch_input.weather_set
        substances = batch_input.substance_list

        queue: List[Tuple[str, Dict[str, Any]]] = []

        if not scenarios:
            return queue

        if batch_input.combinations and (weathers or substances):
            # Create all combinations
            weathers = weathers or [{}]
            substances = substances or [{}]

            combo_idx = 0
            for s_idx, scenario in enumerate(scenarios):
                for w_idx, weather in enumerate(weathers):
                    for su_idx, substance in enumerate(substances):
                        merged = deepcopy(scenario)

                        if isinstance(weather, dict) and weather:
                            # Merge weather as top-level keys
                            for kw, vw in weather.items():
                                merged[kw] = vw

                        if isinstance(substance, dict) and substance:
                            # Merge substance properties as top-level keys
                            for ks, vs in substance.items():
                                merged[ks] = vs

                        scenario_id = (
                            f"s{s_idx}_w{w_idx}_su{su_idx}_c{combo_idx}"
                        )
                        queue.append((scenario_id, merged))
                        combo_idx += 1
        else:
            # 1:1 mapping
            for idx, scenario in enumerate(scenarios):
                merged = dict(scenario)

                # Merge weather if provided (top-level keys)
                if idx < len(weathers):
                    w = weathers[idx]
                    if isinstance(w, dict):
                        for kw, vw in w.items():
                            merged[kw] = vw

                # Merge substance if provided (top-level keys)
                if idx < len(substances):
                    s = substances[idx]
                    if isinstance(s, dict):
                        for ks, vs in s.items():
                            merged[ks] = vs

                scenario_id = f"s{idx}"
                queue.append((scenario_id, merged))

        return queue

    def _run_parallel(
        self,
        queue: List[Tuple[str, Dict[str, Any]]],
        batch_input: BatchInput,
        callback: Optional[Callable],
    ) -> Tuple[
        List[Tuple[str, Dict[str, Any], Any]],
        List[Tuple[str, str]],
        List[Dict[str, Any]],
    ]:
        """Run scenarios using multiprocessing pool."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = min(batch_input.parallel, len(queue), multiprocessing.cpu_count())
        total = len(queue)

        results: List[Tuple[str, Dict[str, Any], Any]] = []
        failures: List[Tuple[str, str]] = []
        summary_rows: List[Dict[str, Any]] = []

        def _worker(scenario_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
            """Worker function for a single scenario."""
            try:
                result = batch_input.model_function(**scenario)
                metrics = _extract_metrics(result)
                # Remove large arrays before serialization
                if hasattr(result, "concentration_grid"):
                    del result.concentration_grid
                if hasattr(result, "ground_concentration"):
                    del result.ground_concentration
                return {
                    "scenario_id": scenario_id,
                    "success": True,
                    "metrics": metrics,
                    "result_dict": (
                        result.to_dict()
                        if hasattr(result, "to_dict")
                        else {"_result_type": type(result).__name__}
                    ),
                }
            except Exception as exc:
                return {
                    "scenario_id": scenario_id,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(_worker, sid, scenario): (idx, sid)
                for idx, (sid, scenario) in enumerate(queue)
            }

            done_count = 0
            for future in as_completed(future_map):
                idx, sid = future_map[future]
                done_count += 1

                try:
                    worker_result = future.result(timeout=300)
                except Exception as exc:
                    err_msg = f"Worker exception: {exc}"
                    failures.append((sid, err_msg))
                    summary_rows.append({"scenario_id": sid, "error": err_msg})
                    if callback:
                        callback(done_count, total, f"FAILED: {sid}")
                    continue

                if worker_result["success"]:
                    results.append(
                        (sid, queue[idx][1], worker_result.get("result_dict", {}))
                    )
                    row = {"scenario_id": sid, **worker_result.get("metrics", {})}
                    summary_rows.append(row)
                    if callback:
                        callback(done_count, total, f"OK: {sid}")
                else:
                    failures.append((sid, worker_result["error"]))
                    summary_rows.append({
                        "scenario_id": sid,
                        "error": worker_result["error"],
                    })
                    if callback:
                        callback(done_count, total, f"FAILED: {sid}")

        return results, failures, summary_rows

    def _compute_statistics(
        self, summary_rows: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute min/max/mean/std for numeric metrics across scenarios."""
        # Collect numeric values per metric
        per_metric: Dict[str, List[float]] = defaultdict(list)

        for row in summary_rows:
            if "error" in row:
                continue
            for key, value in row.items():
                if key == "scenario_id":
                    continue
                if isinstance(value, (int, float, np.floating)):
                    per_metric[key].append(float(value))

        statistics: Dict[str, Dict[str, float]] = {}
        for metric, values in per_metric.items():
            if len(values) < 1:
                continue
            arr = np.array(values, dtype=np.float64)
            statistics[metric] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "count": len(values),
            }

        return statistics


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


def run_batch(
    scenarios: List[Dict[str, Any]],
    model_function: Callable,
    weather_set: Optional[List[Dict[str, Any]]] = None,
    substance_list: Optional[List[Any]] = None,
    combinations: bool = False,
    parallel: int = 0,
    verbose: bool = False,
    callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchResult:
    """Quick one-shot batch runner.

    Args:
        scenarios: List of scenario parameter dicts.
        model_function: Callable(**scenario) → result object.
        weather_set: Optional weather overrides.
        substance_list: Optional substance overrides.
        combinations: If True, run all combinations.
        parallel: Number of parallel workers.
        verbose: Print progress.
        callback: Optional progress callback.

    Returns:
        BatchResult.
    """
    runner = BatchRunner()
    batch_input = BatchInput(
        scenarios=scenarios,
        weather_set=weather_set or [],
        substance_list=substance_list or [],
        combinations=combinations,
        model_function=model_function,
        parallel=parallel,
        verbose=verbose,
    )
    return runner.run(batch_input, callback=callback)
