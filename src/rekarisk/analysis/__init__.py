"""
Rekarisk Advanced Analysis — Batch Runner, Sensitivity & Monte Carlo.

Provides:
    - BatchRunner: run multiple consequence scenarios in sequence
    - Sensitivity: OAT (one-at-a-time) parameter sensitivity analysis
    - Monte Carlo: uncertainty propagation via Monte Carlo simulation
    - Worst Case: worst-case scenario identification from batch results
"""

from __future__ import annotations

from .batch_runner import BatchInput, BatchResult, BatchRunner
from .sensitivity import (
    SensitivityInput,
    SensitivityResult,
    run_oat,
    sensitivity_indices,
    tornado_data,
)
from .monte_carlo import (
    Distribution,
    Normal,
    LogNormal,
    Uniform,
    Triangular,
    Beta,
    MCInput,
    MCResult,
    run_monte_carlo,
    sobol_indices,
    convergence_check,
)
from .worst_case import (
    find_worst_case,
    worst_case_parameters,
)

__all__ = [
    # Batch runner
    "BatchInput",
    "BatchResult",
    "BatchRunner",
    # Sensitivity
    "SensitivityInput",
    "SensitivityResult",
    "run_oat",
    "sensitivity_indices",
    "tornado_data",
    # Monte Carlo
    "Distribution",
    "Normal",
    "LogNormal",
    "Uniform",
    "Triangular",
    "Beta",
    "MCInput",
    "MCResult",
    "run_monte_carlo",
    "sobol_indices",
    "convergence_check",
    # Worst case
    "find_worst_case",
    "worst_case_parameters",
]
