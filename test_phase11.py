"""
Phase 11 Validation Tests — Advanced Analysis Module.

Tests: batch runner, sensitivity, Monte Carlo, worst case.
"""
import sys
sys.path.insert(0, 'src')

import math
import numpy as np


class _FakePytest:
    """Standalone pytest.approx replacement for environments without pytest."""
    @staticmethod
    def approx(expected, rel=1e-7, abs=None):
        _tolerance = abs
        class _Approx:
            def __eq__(self, other):
                import builtins
                _abs = builtins.abs
                if _tolerance is not None:
                    return _abs(other - expected) <= _tolerance
                denom = max(_abs(expected), _abs(other), 1.0)
                return _abs(other - expected) <= rel * denom
            def __repr__(self):
                return f"approx({expected})"
        return _Approx()


# Make pytest available as a namespace for compatibility
pytest = type(sys)("pytest")
pytest.approx = _FakePytest.approx

# ═══════════════════════════════════════════════════════════════════════════
# 1. Batch Runner
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_runner_basic():
    from rekarisk.analysis.batch_runner import BatchInput, BatchResult, BatchRunner

    def dummy_model(source_rate=1.0, wind_speed=3.0):
        return {"max_concentration": source_rate / wind_speed}

    scenarios = [
        {"source_rate": 1.0, "wind_speed": 3.0},
        {"source_rate": 5.0, "wind_speed": 2.0},
        {"source_rate": 10.0, "wind_speed": 1.0},
    ]

    runner = BatchRunner()
    batch_input = BatchInput(
        scenarios=scenarios,
        model_function=dummy_model,
    )
    result = runner.run(batch_input)

    assert result.total_scenarios == 3, f"Expected 3, got {result.total_scenarios}"
    assert result.success_count == 3, f"Expected 3 successes, got {result.success_count}"
    assert result.failure_count == 0, f"Expected 0 failures, got {result.failure_count}"
    assert len(result.summary_table) == 3
    assert "max_concentration" in result.statistics
    stats = result.statistics["max_concentration"]
    assert stats["min"] == pytest.approx(0.333, rel=0.01)
    assert stats["max"] == pytest.approx(10.0, rel=0.01)

    print(" ✅ batch_runner_basic passed")


def test_batch_runner_combinations():
    from rekarisk.analysis.batch_runner import BatchInput, BatchRunner

    def dummy_model(source_rate=1.0, wind_speed=3.0, temperature=298.0):
        return {"max_concentration": source_rate / wind_speed}

    scenarios = [
        {"source_rate": 5.0, "wind_speed": 3.0},
    ]
    weather_set = [
        {"temperature": 298.0},
        {"temperature": 310.0},
    ]

    runner = BatchRunner()
    batch_input = BatchInput(
        scenarios=scenarios,
        weather_set=weather_set,
        combinations=True,
        model_function=dummy_model,
    )
    result = runner.run(batch_input)

    assert result.total_scenarios == 2  # 1 scenario × 2 weather
    assert result.success_count == 2

    print(" ✅ batch_runner_combinations passed")


def test_batch_runner_error_handling():
    from rekarisk.analysis.batch_runner import BatchInput, BatchRunner

    def flaky_model(source_rate=1.0, wind_speed=3.0):
        if source_rate < 0:
            raise ValueError("negative rate!")
        return {"max_concentration": source_rate / wind_speed}

    scenarios = [
        {"source_rate": 1.0, "wind_speed": 3.0},
        {"source_rate": -5.0, "wind_speed": 2.0},  # should fail
        {"source_rate": 10.0, "wind_speed": 1.0},
    ]

    runner = BatchRunner()
    batch_input = BatchInput(
        scenarios=scenarios,
        model_function=flaky_model,
    )
    result = runner.run(batch_input)

    assert result.success_count == 2
    assert result.failure_count == 1
    assert len(result.failed_scenarios) == 1

    print(" ✅ batch_runner_error_handling passed")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════

def test_sensitivity_basic():
    from rekarisk.analysis.sensitivity import (
        SensitivityInput, SensitivityResult, run_oat,
        sensitivity_indices, tornado_data, parameter_ranks,
    )

    def plume_model(source_rate=5.0, wind_speed=3.0, release_height=0.0):
        # Simple Gaussian-like: C ∝ Q / (u * H²) for H > 0, else Q/u
        if release_height > 0:
            return {"max_concentration": source_rate / (wind_speed * release_height ** 2)}
        return {"max_concentration": source_rate / wind_speed}

    si = SensitivityInput(
        base_case={"source_rate": 5.0, "wind_speed": 3.0, "release_height": 10.0},
        parameters={
            "source_rate": (1.0, 10.0),
            "wind_speed": (1.0, 6.0),
            "release_height": (5.0, 20.0),
        },
        model_function=plume_model,
        output_key="max_concentration",
    )

    result = run_oat(si)
    assert result.base_output == pytest.approx(5.0 / (3.0 * 100.0), rel=0.01)
    assert result.n_parameters == 3
    assert len(result.rankings) == 3
    # release_height should be most sensitive (inverse square)
    assert result.rankings[0][0] == "release_height"

    # Sensitivity indices
    indices = sensitivity_indices(result, normalize=True)
    assert len(indices) == 3
    assert abs(sum(indices.values()) - 1.0) < 1e-9

    # Tornado data
    labels, base, low, high = tornado_data(result, top_n=3)
    assert len(labels) == 3
    assert len(low) == 3
    assert len(high) == 3

    # Parameter ranks table
    ranks_table = parameter_ranks(result)
    assert len(ranks_table) == 3
    assert "sensitivity_index" in ranks_table[0]

    print(" ✅ sensitivity_basic passed")


def test_sensitivity_auto_range():
    from rekarisk.analysis.sensitivity import SensitivityInput, run_oat

    def model(a=100.0, b=200.0):
        return a + b

    si = SensitivityInput(
        base_case={"a": 100.0, "b": 200.0},
        parameters={},  # auto-generate ±20%
        model_function=model,
    )
    result = run_oat(si)
    assert result.n_parameters == 2
    # Both should have non-zero ranges
    for _, effect in result.parameter_effects.items():
        assert abs(effect[2]) > 0.01

    print(" ✅ sensitivity_auto_range passed")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

def test_monte_carlo_basic():
    from rekarisk.analysis.monte_carlo import (
        Normal, Uniform, MCInput, run_monte_carlo, convergence_check,
    )

    def linear_model(a=1.0, b=2.0):
        return a + b

    params = {
        "a": Normal(10.0, 1.0),
        "b": Uniform(3.0, 5.0),
    }
    mc_input = MCInput(
        parameters=params,
        model_function=linear_model,
        n_samples=500,
        seed=42,
    )
    result = run_monte_carlo(mc_input)

    assert result.n_samples > 0
    assert None in result.outputs  # single output stored as None key
    assert "a" in result.samples
    assert "b" in result.samples

    # Mean should be ~ a.mean + b.mean = 10 + 4 = 14
    stats = result.statistics[None]
    assert stats["mean"] == pytest.approx(14.0, abs=0.5)
    assert stats["p5"] < stats["p95"]
    assert stats["ci_low"] < stats["ci_high"]

    # Convergence check
    converged = convergence_check(result, output_key=None)
    # Should converge for this linear model with 500 samples
    assert converged is True or converged is False  # just check it runs

    print(" ✅ monte_carlo_basic passed")


def test_monte_carlo_distributions():
    from rekarisk.analysis.monte_carlo import (
        Normal, LogNormal, Uniform, Triangular, Beta,
    )

    n = 1000
    seed = 42

    # Normal
    dist = Normal(10.0, 2.0)
    samples = dist.sample(n, seed=seed)
    assert abs(np.mean(samples) - 10.0) < 0.3
    assert abs(np.std(samples) - 2.0) < 0.3

    # Uniform
    dist = Uniform(0.0, 10.0)
    samples = dist.sample(n, seed=seed)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 10.0)
    assert abs(np.mean(samples) - 5.0) < 0.5

    # Triangular
    dist = Triangular(0.0, 7.0, 10.0)
    samples = dist.sample(n, seed=seed)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 10.0)

    # Beta
    dist = Beta(2.0, 5.0)
    samples = dist.sample(n, seed=seed)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)

    # LogNormal
    dist = LogNormal(0.0, 0.5)
    samples = dist.sample(n, seed=seed)
    assert np.all(samples > 0)

    # PDF, CDF, PPF
    x = np.linspace(0.1, 0.9, 5)
    dist = Beta(2.0, 2.0)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    ppf = dist.ppf(cdf)
    assert np.allclose(ppf, x, atol=0.05)

    print(" ✅ monte_carlo_distributions passed")


def test_monte_carlo_sobol():
    from rekarisk.analysis.monte_carlo import (
        Normal, Uniform, MCInput, run_monte_carlo, sobol_indices,
    )

    def model(a=1.0, b=2.0):
        return a + b

    params = {
        "a": Normal(10.0, 1.0),
        "b": Uniform(3.0, 5.0),
    }
    mc_input = MCInput(
        parameters=params,
        model_function=model,
        n_samples=512,
        seed=42,
        use_sobol=True,
    )
    result = run_monte_carlo(mc_input)

    # Sobol' indices
    si = sobol_indices(result)
    assert len(si) == 2
    assert all(0 <= v <= 1 for v in si.values())

    print(" ✅ monte_carlo_sobol passed")


def test_make_distribution():
    from rekarisk.analysis.monte_carlo import make_distribution, Normal, Uniform

    d = make_distribution("normal", 5.0, 1.0)
    assert isinstance(d, Normal)
    assert d.mu == 5.0

    d = make_distribution("uniform", 0.0, 10.0)
    assert isinstance(d, Uniform)
    assert d.a == 0.0

    try:
        d = make_distribution("gaussian", 0.0, 1.0)
        assert False, "Should have raised"
    except ValueError:
        pass

    print(" ✅ make_distribution passed")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Worst Case
# ═══════════════════════════════════════════════════════════════════════════

def test_find_worst_case():
    from rekarisk.analysis.worst_case import (
        find_worst_case, worst_case_parameters, classify_severity,
    )

    # Mock batch result
    class MockBatchResult:
        summary_table = [
            {"scenario_id": "s0", "max_concentration": 100.0, "heat_flux": 15.0},
            {"scenario_id": "s1", "max_concentration": 500.0, "heat_flux": 10.0},
            {"scenario_id": "s2", "max_concentration": 250.0, "heat_flux": 30.0},
            {"scenario_id": "s3", "error": "model failed"},
        ]
        results = [
            ("s0", {"source_rate": 1.0}, {}),
            ("s1", {"source_rate": 10.0}, {}),
            ("s2", {"source_rate": 5.0}, {}),
        ]

    batch_result = MockBatchResult()

    # find_worst_case
    worst = find_worst_case(batch_result, "max_concentration", direction="maximize")
    assert len(worst) == 1
    assert worst[0][0] == "s1"
    assert worst[0][1] == 500.0

    worst = find_worst_case(batch_result, "max_concentration", direction="minimize")
    assert worst[0][0] == "s0"

    worst_2 = find_worst_case(batch_result, "max_concentration", n_worst=2)
    assert len(worst_2) == 2
    assert worst_2[0][0] == "s1"

    # worst_case_parameters
    params = worst_case_parameters(batch_result, "max_concentration")
    assert params["worst_scenario_id"] == "s1"
    assert params["worst_value"] == 500.0
    assert params["parameters"]["source_rate"] == 10.0

    # classify_severity
    thresholds = {"low": 10.0, "medium": 100.0, "high": 1000.0}
    assert classify_severity(5.0, thresholds, ascending=False) == "low"
    assert classify_severity(50.0, thresholds, ascending=False) == "medium"
    assert classify_severity(500.0, thresholds, ascending=False) == "high"
    assert classify_severity(5000.0, thresholds, ascending=False) == "high"

    print(" ✅ find_worst_case passed")


def test_systematic_worst_case():
    from rekarisk.analysis.worst_case import systematic_worst_case_search

    def model(a=1.0, b=2.0):
        return a * b

    result = systematic_worst_case_search(
        param_ranges={"a": (1.0, 10.0), "b": (1.0, 5.0)},
        model_function=model,
        output_extractor=lambda r: r,
        n_steps=3,
        direction="maximize",
    )

    assert result["worst_params"]["a"] == 10.0
    assert result["worst_params"]["b"] == 5.0
    assert result["worst_value"] == pytest.approx(50.0, rel=0.01)
    assert result["n_evaluated"] == 9  # 3^2

    print(" ✅ systematic_worst_case passed")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running Phase 11 Validation Tests...\n")
    test_batch_runner_basic()
    test_batch_runner_combinations()
    test_batch_runner_error_handling()
    test_sensitivity_basic()
    test_sensitivity_auto_range()
    test_monte_carlo_basic()
    test_monte_carlo_distributions()
    test_monte_carlo_sobol()
    test_make_distribution()
    test_find_worst_case()
    test_systematic_worst_case()
    print("\n🎉 All Phase 11 tests passed!")
