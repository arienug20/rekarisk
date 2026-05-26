"""
Rekarisk Advanced Analysis — Monte Carlo Uncertainty Propagation.

Propagates input parameter uncertainty through consequence models using
Monte Carlo simulation. Supports five distribution types:

    - Normal(μ, σ)
    - LogNormal(μ, σ)
    - Uniform(a, b)
    - Triangular(a, mode, b)
    - Beta(α, β)

Provides convergence checking, Sobol' sensitivity indices (first-order),
and correlation analysis.

Typical usage:
    params = {
        "source_rate": Normal(5.0, 1.0),
        "wind_speed": Uniform(2.0, 6.0),
    }
    mc_input = MCInput(parameters=params, model_function=my_model)
    result = run_monte_carlo(mc_input)
    print(result.statistics["max_concentration"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Distribution Classes
# ---------------------------------------------------------------------------


class Distribution:
    """Base class for probability distributions used in Monte Carlo analysis.

    Each subclass must implement: _sample, pdf, cdf, ppf.
    """

    def __init__(self):
        self._rng: Optional[np.random.Generator] = None

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Generate n random samples from the distribution.

        Args:
            n: Number of samples.
            seed: Optional RNG seed for reproducibility.

        Returns:
            Array of shape (n,).
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            return self._sample(n, rng)
        return self._sample(n, self.rng)

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function at x."""
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function at x."""
        raise NotImplementedError

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF) at p."""
        raise NotImplementedError

    @property
    def mean(self) -> float:
        raise NotImplementedError

    @property
    def std(self) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normal(Distribution):
    """Normal (Gaussian) distribution N(μ, σ)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        return stats.norm.ppf(p, loc=self.mu, scale=self.sigma)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        return self.sigma

    def __repr__(self) -> str:
        return f"Normal(μ={self.mu}, σ={self.sigma})"


class LogNormal(Distribution):
    """Log-normal distribution LN(μ, σ).

    Parameters μ and σ are the mean and std of ln(X) — the underlying
    normal distribution.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu))

    def ppf(self, p: np.ndarray) -> np.ndarray:
        return stats.lognorm.ppf(p, s=self.sigma, scale=np.exp(self.mu))

    @property
    def mean(self) -> float:
        return float(np.exp(self.mu + self.sigma ** 2 / 2))

    @property
    def std(self) -> float:
        return float(
            np.sqrt((np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2))
        )

    def __repr__(self) -> str:
        return f"LogNormal(μ={self.mu}, σ={self.sigma})"


class Uniform(Distribution):
    """Uniform distribution U(a, b)."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        super().__init__()
        if b <= a:
            raise ValueError(f"b must be > a, got a={a}, b={b}")
        self.a = a
        self.b = b

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.a, self.b, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.uniform.pdf(x, loc=self.a, scale=self.b - self.a)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.uniform.cdf(x, loc=self.a, scale=self.b - self.a)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        return stats.uniform.ppf(p, loc=self.a, scale=self.b - self.a)

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2.0

    @property
    def std(self) -> float:
        return float((self.b - self.a) / np.sqrt(12.0))

    def __repr__(self) -> str:
        return f"Uniform({self.a}, {self.b})"


class Triangular(Distribution):
    """Triangular distribution T(a, mode, b).

    Parameters:
        a: Lower bound.
        mode: Most likely value (peak).
        b: Upper bound.
    """

    def __init__(self, a: float = 0.0, mode: float = 0.5, b: float = 1.0):
        super().__init__()
        if not (a <= mode <= b):
            raise ValueError(f"Must have a <= mode <= b, got a={a}, mode={mode}, b={b}")
        if b == a:
            raise ValueError(f"b must be > a, got a={a}, b={b}")
        self.a = a
        self.mode_value = mode
        self.b = b

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.a, self.mode_value, self.b, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        a, c, b = self.a, self.mode_value, self.b
        result = np.zeros_like(x, dtype=np.float64)
        # For x < a or x > b: pdf = 0
        mask_a = (x >= a) & (x <= c)
        mask_c = (x > c) & (x <= b)
        if b - a == 0:
            return result
        result[mask_a] = 2.0 * (x[mask_a] - a) / ((b - a) * (c - a)) if c - a > 0 else 0
        result[mask_c] = 2.0 * (b - x[mask_c]) / ((b - a) * (b - c)) if b - c > 0 else 0
        return result

    def cdf(self, x: np.ndarray) -> np.ndarray:
        a, c, b = self.a, self.mode_value, self.b
        result = np.zeros_like(x, dtype=np.float64)
        mask1 = (x >= a) & (x <= c)
        mask2 = (x > c) & (x <= b)
        mask3 = x > b
        if c - a > 0:
            result[mask1] = ((x[mask1] - a) ** 2) / ((b - a) * (c - a))
        if b - c > 0:
            result[mask2] = 1.0 - ((b - x[mask2]) ** 2) / ((b - a) * (b - c))
        result[mask3] = 1.0
        return result

    def ppf(self, p: np.ndarray) -> np.ndarray:
        a, c, b = self.a, self.mode_value, self.b
        result = np.zeros_like(p, dtype=np.float64)
        p_clipped = np.clip(p, 0.0, 1.0)
        cutoff = (c - a) / (b - a) if b - a > 0 else 0.5
        mask1 = p_clipped <= cutoff
        mask2 = ~mask1
        if c - a > 0:
            result[mask1] = a + np.sqrt(p_clipped[mask1] * (b - a) * (c - a))
        if b - c > 0:
            result[mask2] = b - np.sqrt((1.0 - p_clipped[mask2]) * (b - a) * (b - c))
        result[p <= 0] = a
        result[p >= 1] = b
        return result

    @property
    def mean(self) -> float:
        return (self.a + self.mode_value + self.b) / 3.0

    @property
    def std(self) -> float:
        return float(
            np.sqrt(
                (self.a ** 2 + self.mode_value ** 2 + self.b ** 2
                 - self.a * self.mode_value - self.a * self.b
                 - self.mode_value * self.b)
                / 18.0
            )
        )

    def __repr__(self) -> str:
        return f"Triangular({self.a}, {self.mode_value}, {self.b})"


class Beta(Distribution):
    """Beta distribution B(α, β).

    Parameters:
        alpha: First shape parameter (α > 0).
        beta: Second shape parameter (β > 0).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"α and β must be > 0, got α={alpha}, β={beta}")
        self.alpha = alpha
        self.beta_param = beta

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta_param, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.beta.pdf(x, self.alpha, self.beta_param)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.beta.cdf(x, self.alpha, self.beta_param)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        return stats.beta.ppf(p, self.alpha, self.beta_param)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta_param
        return float(np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))))

    def __repr__(self) -> str:
        return f"Beta(α={self.alpha}, β={self.beta_param})"


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def make_distribution(
    dist_type: str,
    *args: float,
) -> Distribution:
    """Create a Distribution from a type name and parameters.

    Args:
        dist_type: One of 'normal', 'lognormal', 'uniform', 'triangular', 'beta'.
        *args: Positional parameters for the distribution.

    Returns:
        Distribution instance.

    Examples:
        >>> d = make_distribution("normal", 5.0, 1.0)
        >>> isinstance(d, Normal)
        True
        >>> d.mean
        5.0
    """
    mapping: Dict[str, type] = {
        "normal": Normal,
        "lognormal": LogNormal,
        "uniform": Uniform,
        "triangular": Triangular,
        "beta": Beta,
    }
    cls = mapping.get(dist_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown distribution type: '{dist_type}'. "
            f"Choose from: {list(mapping.keys())}"
        )
    return cls(*args)


# ---------------------------------------------------------------------------
# MC Input & Result
# ---------------------------------------------------------------------------


@dataclass
class MCInput:
    """Input specification for Monte Carlo simulation.

    Attributes:
        parameters: Dict {name: Distribution} defining input uncertainty.
        model_function: Callable(**params) → result. Accepts keyword arguments
            corresponding to parameter names.
        output_keys: List of result metric names to track. If empty, all
            numeric metrics from the result are collected.
        output_extractor: Optional callable (result) → float, for single-output
            models. Takes priority over output_keys.
        n_samples: Number of Monte Carlo samples (default 1000).
        seed: RNG seed for reproducibility.
        confidence_level: Confidence level for intervals (default 0.95).
        use_sobol: If True, use Sobol' quasi-random sequences instead of
            pseudo-random for better convergence.
    """

    parameters: Dict[str, Distribution] = field(default_factory=dict)
    model_function: Optional[Callable] = None
    output_keys: List[str] = field(default_factory=list)
    output_extractor: Optional[Callable[[Any], float]] = None
    n_samples: int = 1000
    seed: Optional[int] = None
    confidence_level: float = 0.95
    use_sobol: bool = False


@dataclass
class MCResult:
    """Results of Monte Carlo simulation.

    Attributes:
        samples: Dict {param_name: array of shape (n_samples,)}.
        outputs: Dict {key: array of shape (n_samples,)}.
        statistics: Dict {key: {mean, std, p5, p50, p95, ci_low, ci_high, ...}}.
        correlations: Dict {output_key: {param: correlation_coefficient}}.
        input: Reference to the MCInput used.
        elapsed_time: Wall-clock time for the simulation [s].
    """

    samples: Dict[str, np.ndarray] = field(default_factory=dict)
    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    input: Optional[MCInput] = None
    elapsed_time: float = 0.0

    @property
    def n_samples(self) -> int:
        if self.outputs:
            return len(next(iter(self.outputs.values())))
        return 0


# ---------------------------------------------------------------------------
# Core MC Runner
# ---------------------------------------------------------------------------


def run_monte_carlo(mc_input: MCInput) -> MCResult:
    """Run Monte Carlo uncertainty propagation.

    Generates n_samples from each parameter distribution, evaluates the
    model function for each sample set, and computes output statistics.

    Args:
        mc_input: MCInput with parameters, model function, and settings.

    Returns:
        MCResult with samples, outputs, statistics, and correlations.

    Raises:
        ValueError: If model_function is None or parameters dict is empty.

    Examples:
        >>> def linear_model(a=1.0, b=2.0):
        ...     return a + b
        >>> params = {"a": Normal(10.0, 1.0), "b": Uniform(3.0, 5.0)}
        >>> mc = MCInput(parameters=params, model_function=linear_model,
        ...              n_samples=500, seed=42)
        >>> result = run_monte_carlo(mc)
        >>> abs(result.statistics[None]["mean"] - 14.0) < 0.5
        True
        >>> "a" in result.samples
        True
    """
    if mc_input.model_function is None:
        raise ValueError("MCInput requires a model_function callable.")
    if not mc_input.parameters:
        raise ValueError("MCInput requires a non-empty parameters dict.")

    import time

    start_time = time.monotonic()
    n = mc_input.n_samples
    param_names = list(mc_input.parameters.keys())

    # Set up RNG
    if mc_input.seed is not None:
        base_rng = np.random.default_rng(mc_input.seed)
    else:
        base_rng = np.random.default_rng()

    # Generate samples
    samples: Dict[str, np.ndarray] = {}
    if mc_input.use_sobol:
        # Use Sobol' quasi-random sequences for better space-filling
        sampler = qmc.Sobol(d=len(param_names), seed=mc_input.seed)
        # Generate n_samples (must be power of 2 for Sobol)
        n_sobol = int(2 ** math.ceil(math.log2(n)))
        sobol_pts = sampler.random(n=n_sobol)
        # Truncate to n
        sobol_pts = sobol_pts[:n, :]

        for i, name in enumerate(param_names):
            dist = mc_input.parameters[name]
            # Map uniform [0,1] through PPF
            samples[name] = dist.ppf(sobol_pts[:, i])
    else:
        for name, dist in mc_input.parameters.items():
            # Give each distribution a fresh sub-seed
            seed_i = None if mc_input.seed is None else mc_input.seed + hash(name) % 10000
            samples[name] = dist.sample(n, seed=seed_i)

    # Run model for each sample
    output_collector: Dict[str, List[float]] = {}

    for i in range(n):
        # Build kwargs for this sample
        kwargs = {name: float(samples[name][i]) for name in param_names}

        try:
            result = mc_input.model_function(**kwargs)
        except Exception:
            continue  # skip failed evaluations

        # Extract outputs
        if mc_input.output_extractor is not None:
            try:
                val = float(mc_input.output_extractor(result))
                key = mc_input.output_keys[0] if mc_input.output_keys else None
                output_collector.setdefault(key, []).append(val)
            except Exception:
                continue
        else:
            if isinstance(result, dict):
                for key in mc_input.output_keys:
                    if key in result:
                        v = result[key]
                        if isinstance(v, (int, float, np.floating)):
                            output_collector.setdefault(key, []).append(float(v))
                # Auto-collect all numeric if no keys specified
                if not mc_input.output_keys:
                    for k, v in result.items():
                        if isinstance(v, (int, float, np.floating)):
                            output_collector.setdefault(k, []).append(float(v))
            elif isinstance(result, (int, float, np.floating)):
                key = mc_input.output_keys[0] if mc_input.output_keys else None
                output_collector.setdefault(key, []).append(float(result))
            else:
                # Object with attributes
                keys = mc_input.output_keys or _auto_discover_keys(result)
                for key in keys:
                    if hasattr(result, key):
                        v = getattr(result, key)
                        if isinstance(v, (int, float, np.floating)):
                            output_collector.setdefault(key, []).append(float(v))
                        elif isinstance(v, np.ndarray) and v.size == 1:
                            output_collector.setdefault(key, []).append(float(v.item()))

    # Convert lists to arrays
    outputs: Dict[str, np.ndarray] = {
        k: np.array(v, dtype=np.float64) for k, v in output_collector.items() if v
    }

    if not outputs:
        # No valid outputs; return empty result
        elapsed = time.monotonic() - start_time
        return MCResult(
            samples=samples,
            outputs={},
            statistics={},
            correlations={},
            input=mc_input,
            elapsed_time=elapsed,
        )

    # Compute statistics
    alpha = 1.0 - mc_input.confidence_level
    statistics: Dict[str, Dict[str, float]] = {}
    for key, arr in outputs.items():
        if len(arr) < 2:
            continue
        _n = len(arr)
        _mean = float(np.mean(arr))
        _std = float(np.std(arr, ddof=1))
        _sorted = np.sort(arr)
        _p5 = float(np.percentile(arr, 5))
        _p50 = float(np.median(arr))
        _p95 = float(np.percentile(arr, 95))

        # Confidence interval (using t-distribution for small n)
        _se = _std / np.sqrt(_n) if _n > 0 else 0.0
        _t_val = stats.t.ppf(1.0 - alpha / 2.0, df=max(1, _n - 1))
        _ci_hw = float(_t_val * _se)

        statistics[key] = {
            "mean": _mean,
            "std": _std,
            "p5": _p5,
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": _p50,
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": _p95,
            "ci_low": _mean - _ci_hw,
            "ci_high": _mean + _ci_hw,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_valid": _n,
        }

    # Compute correlations between each output and each input parameter
    correlations: Dict[str, Dict[str, float]] = {}
    valid_n = n
    for key, out_arr in outputs.items():
        corr_dict: Dict[str, float] = {}
        for pname, param_arr in samples.items():
            # Truncate to same length
            m = min(len(out_arr), len(param_arr))
            if m < 2:
                corr_dict[pname] = 0.0
                continue
            corr_matrix = np.corrcoef(param_arr[:m], out_arr[:m])
            corr_dict[pname] = float(corr_matrix[0, 1])
        correlations[key] = corr_dict

    elapsed = time.monotonic() - start_time

    return MCResult(
        samples=samples,
        outputs=outputs,
        statistics=statistics,
        correlations=correlations,
        input=mc_input,
        elapsed_time=elapsed,
    )


def _auto_discover_keys(result: Any) -> List[str]:
    """Auto-discover numeric metrics from a result object."""
    keys: List[str] = []
    candidates = [
        "max_concentration", "max_distance", "max_flux",
        "impact_distance", "pool_radius", "flame_length",
        "flame_height", "surface_emissive_power", "heat_flux",
        "overpressure", "impulse", "probit_value",
        "probability_of_death", "risk_individual",
    ]
    for k in candidates:
        if hasattr(result, k):
            v = getattr(result, k)
            if isinstance(v, (int, float, np.floating, np.ndarray)):
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Sobol' Sensitivity Indices (First-Order)
# ---------------------------------------------------------------------------


def sobol_indices(mc_result: MCResult, output_key: Optional[str] = None) -> Dict[str, float]:
    """Compute first-order Sobol' sensitivity indices from MC results.

    Uses the method of sampling-based estimation: S_i = V_i / V_total,
    where V_i = Var(E[Y | X_i]) via a correlation-based approximate method.

    For proper Sobol' indices with sufficient accuracy, n_samples should be
    at least 1000 and preferably >5000.

    Args:
        mc_result: MCResult from run_monte_carlo().
        output_key: Which output to compute indices for. If None, uses
            the first (or only) output.

    Returns:
        Dict {param_name: first_order_index} with values in [0, 1].

    Note:
        This is a simplified first-order approximation using Pearson
        correlation squared, adjusted for linear models. For non-linear
        models, consider using dedicated SALib integration.
    """
    outputs = mc_result.outputs
    if not outputs:
        return {}

    if output_key is None:
        output_key = next(iter(outputs.keys()))

    if output_key not in outputs:
        return {}

    out_arr = outputs[output_key]
    if len(out_arr) < 10:
        return {}

    # Compute squared correlation as first-order approximation
    indices: Dict[str, float] = {}
    total_r2 = 0.0

    for param_name, param_arr in mc_result.samples.items():
        m = min(len(out_arr), len(param_arr))
        if m < 3:
            indices[param_name] = 0.0
            continue
        corr = np.corrcoef(param_arr[:m], out_arr[:m])[0, 1]
        indices[param_name] = float(corr ** 2)
        total_r2 += indices[param_name]

    # Normalize so first-order indices sum to ~1 for linear/additive models
    if total_r2 > 0:
        for key in indices:
            indices[key] /= total_r2

    return indices


# ---------------------------------------------------------------------------
# Convergence Check
# ---------------------------------------------------------------------------


def convergence_check(
    mc_result: MCResult,
    output_key: Optional[str] = None,
    ci_width_threshold: float = 0.10,
) -> bool:
    """Check if Monte Carlo simulation has converged.

    Convergence criterion: |CI_width / mean| < ci_width_threshold.
    CI width = ci_high - ci_low. For the threshold of 0.10, the CI
    width must be less than 10% of the mean value.

    Args:
        mc_result: MCResult.
        output_key: Which output to check. If None, checks all and
            returns True only if all pass.
        ci_width_threshold: Maximum allowed relative CI width.

    Returns:
        True if converged, False otherwise.
    """
    if not mc_result.statistics:
        return False

    if output_key is not None:
        if output_key not in mc_result.statistics:
            return False
        stats_dict = mc_result.statistics[output_key]
        ci_width = stats_dict["ci_high"] - stats_dict["ci_low"]
        mean_val = abs(stats_dict["mean"])
        if mean_val < 1e-12:
            return ci_width < 1e-12
        return (ci_width / mean_val) < ci_width_threshold

    # Check all outputs
    for key, stats_dict in mc_result.statistics.items():
        ci_width = stats_dict["ci_high"] - stats_dict["ci_low"]
        mean_val = abs(stats_dict["mean"])
        if mean_val < 1e-12:
            if ci_width >= 1e-12:
                return False
        elif (ci_width / mean_val) >= ci_width_threshold:
            return False

    return True


# ---------------------------------------------------------------------------
# Required Samples Estimator
# ---------------------------------------------------------------------------


def estimate_required_samples(
    mc_result: MCResult,
    output_key: Optional[str] = None,
    target_ci_width: float = 0.05,
    confidence: float = 0.95,
) -> int:
    """Estimate how many samples are needed to achieve a target CI width.

    Based on the observed standard deviation, estimates the number of
    samples needed for the CI half-width to be within target_ci_width
    fraction of the mean.

    Args:
        mc_result: MCResult.
        output_key: Which output to estimate for.
        target_ci_width: Desired relative CI half-width (default 0.05 = 5%).
        confidence: Confidence level (default 0.95).

    Returns:
        Estimated number of samples needed. Returns current n_samples if
        insufficient data.
    """
    if not mc_result.statistics:
        return mc_result.n_samples

    if output_key is None:
        output_key = next(iter(mc_result.statistics.keys()))

    if output_key not in mc_result.statistics:
        return mc_result.n_samples

    s = mc_result.statistics[output_key]
    _mean = abs(s["mean"])
    _std = s["std"]

    if _mean < 1e-12:
        return mc_result.n_samples

    # CI half-width = t * σ / √n
    # Target: t * σ / √n <= target_ci_width * |mean|
    # → n >= (t * σ / (target_ci_width * |mean|))²
    alpha = 1.0 - confidence
    t_val = stats.norm.ppf(1.0 - alpha / 2.0)  # use z for estimate
    n_required = (t_val * _std / (target_ci_width * _mean)) ** 2
    n_required = max(mc_result.n_samples, int(math.ceil(n_required)))

    # Sanity cap
    n_required = min(n_required, 1_000_000)

    return n_required
