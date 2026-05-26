"""
Rekarisk — Phase Equilibrium & Flash Calculations.

Implements PT Flash, bubble/dew point calculations, and phase envelope
construction using cubic equations of state (Peng-Robinson).

References:
  - Michelsen, M.L. (1982). "The Isothermal Flash Problem", Fluid Phase Equilib.
  - Rachford, H.H. & Rice, J.D. (1952). JPT, 4(10), 327.
  - Whitson, C.H. & Brule, M.R. "Phase Behavior", SPE Monograph.

All calculations in SI units: T [K], P [Pa], compositions as mole fractions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import R, EPSILON, T_0C, P_ATM
from .eos import (
    CubicEoS, PengRobinson, SoaveRedlichKwong,
    EoSParameters, create_mixture_eos
)


# ══════════════════════════════════════════════════════════════════════════════
# Flash Calculation Data Structures
# ══════════════════════════════════════════════════════════════════════════════

class FlashResult:
    """Result of a PT flash calculation."""

    def __init__(self):
        self.beta: float = 0.0          # Vapor fraction (0=all liquid, 1=all vapor)
        self.T: float = 300.0           # Temperature [K]
        self.P: float = 1e5             # Pressure [Pa]
        self.x: np.ndarray = np.array([])  # Liquid mole fractions
        self.y: np.ndarray = np.array([])  # Vapor mole fractions
        self.z: np.ndarray = np.array([])  # Feed mole fractions
        self.K_values: np.ndarray = np.array([])  # K-values (y/x)
        self.iterations: int = 0        # Outer iterations
        self.converged: bool = False
        self.message: str = ""

    @property
    def is_two_phase(self) -> bool:
        """True if 0 < beta < 1 (within tolerance)."""
        return 1e-8 < self.beta < 1.0 - 1e-8

    @property
    def is_liquid(self) -> bool:
        """True if beta = 0 (subcooled liquid)."""
        return self.beta <= 1e-8

    @property
    def is_vapor(self) -> bool:
        """True if beta = 1 (superheated vapor)."""
        return self.beta >= 1.0 - 1e-8

    def to_dict(self) -> Dict:
        return {
            'beta': self.beta,
            'T': self.T,
            'P': self.P,
            'x': self.x.tolist() if len(self.x) > 0 else [],
            'y': self.y.tolist() if len(self.y) > 0 else [],
            'z': self.z.tolist() if len(self.z) > 0 else [],
            'K_values': self.K_values.tolist() if len(self.K_values) > 0 else [],
            'iterations': self.iterations,
            'converged': self.converged,
            'message': self.message,
            'is_two_phase': self.is_two_phase,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Rachford-Rice Solver
# ══════════════════════════════════════════════════════════════════════════════

def rachford_rice(beta: float, K: np.ndarray, z: np.ndarray) -> float:
    """Rachford-Rice equation f(beta) = 0.

    f(beta) = Σ_i [z_i·(K_i - 1) / (1 + beta·(K_i - 1))]

    Args:
        beta: Vapor fraction (0 < beta < 1).
        K: K-values array (K_i = y_i/x_i).
        z: Feed mole fractions.

    Returns:
        f(beta) value (should be 0 at solution).
    """
    K_minus_1 = K - 1.0
    denom = 1.0 + beta * K_minus_1
    # Guard against division by zero
    mask = np.abs(denom) > EPSILON
    result = 0.0
    for i in range(len(z)):
        if mask[i]:
            result += z[i] * K_minus_1[i] / denom[i]
    return result


def rachford_rice_derivative(beta: float, K: np.ndarray, z: np.ndarray) -> float:
    """Derivative of Rachford-Rice: f'(beta).

    f'(beta) = -Σ_i [z_i·(K_i - 1)² / (1 + beta·(K_i - 1))²]
    """
    K_minus_1 = K - 1.0
    denom = 1.0 + beta * K_minus_1
    result = 0.0
    for i in range(len(z)):
        if abs(denom[i]) > EPSILON:
            result -= z[i] * K_minus_1[i] ** 2 / (denom[i] ** 2)
    return result


def solve_rachford_rice(K: np.ndarray, z: np.ndarray,
                         max_iter: int = 100,
                         tol: float = 1e-10) -> Tuple[float, bool, int]:
    """Solve for vapor fraction beta using Newton-Raphson on Rachford-Rice.

    Handles edge cases: beta = 0 (all liquid) and beta = 1 (all vapor).

    Args:
        K: K-values array.
        z: Feed mole fractions.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (beta, converged, iterations)
    """
    n = len(z)

    # Check if flash is possible
    # At beta=0: f(0) = Σ z_i·(K_i - 1)
    f0 = np.sum(z * (K - 1.0))
    # At beta=1: f(1) = Σ z_i·(K_i - 1)/K_i = Σ z_i·(1 - 1/K_i)
    f1 = np.sum(z * (1.0 - 1.0 / np.maximum(K, EPSILON)))

    if f0 <= 0:
        # All liquid
        return (0.0, True, 0)
    if f1 >= 0:
        # All vapor
        return (1.0, True, 0)

    # Two-phase: solve for beta in (0, 1)
    # Initial guess
    beta = 0.5
    # Better initial guess: approximate from f0 and f1
    if abs(f0 - f1) > EPSILON:
        beta = max(min(-f0 / (f1 - f0), 0.99), 0.01)

    converged = False
    for it in range(max_iter):
        f_val = rachford_rice(beta, K, z)
        if abs(f_val) < tol:
            converged = True
            break

        df = rachford_rice_derivative(beta, K, z)
        if abs(df) < EPSILON:
            # Bisection step
            if f_val > 0:
                beta = (beta + 1.0) / 2.0
            else:
                beta = beta / 2.0
        else:
            delta = f_val / df
            # Damp if step is too large
            if abs(delta) > 0.5:
                delta = math.copysign(0.5, delta)
            beta -= delta

        # Bound beta to (0, 1)
        beta = max(min(beta, 1.0 - 1e-12), 1e-12)

    return (beta, converged, it + 1)


# ══════════════════════════════════════════════════════════════════════════════
# PT Flash — Main Algorithm
# ══════════════════════════════════════════════════════════════════════════════

def pt_flash(P: float, T: float, z: np.ndarray,
             eos: Optional[CubicEoS] = None,
             comp_params: Optional[List[EoSParameters]] = None,
             k_ij: Optional[np.ndarray] = None,
             max_outer: int = 50,
             tol_outer: float = 1e-8,
             tol_inner: float = 1e-10) -> FlashResult:
    """Isothermal-isobaric (PT) flash calculation.

    Given feed composition z, pressure P, and temperature T, determines:
      - Vapor fraction beta
      - Vapor composition y
      - Liquid composition x

    Algorithm:
      1. Initialize K-values (Wilson's correlation)
      2. Solve Rachford-Rice for beta
      3. Compute x and y from beta and K
      4. Compute fugacity coefficients for both phases
      5. Update K-values: K_i = φ_i^L / φ_i^V
      6. Repeat until convergence

    Args:
        P: Pressure [Pa].
        T: Temperature [K].
        z: Feed mole fractions (sum = 1).
        eos: EoS instance (PengRobinson default if None).
        comp_params: List of EoSParameters for each component.
        k_ij: Binary interaction matrix.
        max_outer: Maximum outer iterations.
        tol_outer: Convergence tolerance for K-value update.
        tol_inner: Tolerance for Rachford-Rice solver.

    Returns:
        FlashResult with equilibrium phase compositions.
    """
    n = len(z)

    result = FlashResult()
    result.z = z.copy()
    result.T = T
    result.P = P

    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    # Ensure we have component parameters
    if comp_params is None:
        # Pure component mode: use eos built-in params
        if n == 1:
            comp_params = [eos._params]
        else:
            result.message = "Component parameters required for mixture flash"
            return result

    # Initialize K-values using Wilson's correlation
    # K_i = (Pc_i/P) · exp(5.373·(1 + ω_i)·(1 - Tc_i/T))
    K = np.ones(n)
    for i, cp in enumerate(comp_params):
        if cp.tc > 0 and cp.pc > 0:
            wilson_factor = 5.373 * (1.0 + cp.omega) * (1.0 - cp.tc / T)
            K[i] = (cp.pc / P) * math.exp(wilson_factor)
        K[i] = max(K[i], EPSILON)

    beta = 0.5
    converged = False

    for outer in range(max_outer):
        # Step 2: Solve Rachford-Rice
        beta, rr_converged, inner_iter = solve_rachford_rice(K, z, tol=tol_inner)

        if not rr_converged:
            result.message = f"Rachford-Rice did not converge after {inner_iter} iterations"
            break

        # Step 3: Compute x and y
        K_minus_1 = K - 1.0
        denom = 1.0 + beta * K_minus_1

        x = np.zeros(n)
        y = np.zeros(n)

        for i in range(n):
            if abs(denom[i]) > EPSILON:
                x[i] = z[i] / denom[i]
                y[i] = K[i] * x[i]

        # Normalize
        x_sum = x.sum()
        y_sum = y.sum()
        if x_sum > EPSILON:
            x /= x_sum
        if y_sum > EPSILON:
            y /= y_sum

        # Handle single phase
        if beta <= 1e-8:
            # All liquid
            x = z.copy()
            y = z * K
            y_sum = y.sum()
            if y_sum > EPSILON:
                y /= y_sum
            result.beta = 0.0
            result.x = x
            result.y = y
            result.K_values = K
            result.converged = True
            result.iterations = outer + 1
            result.message = "Single phase liquid"
            return result

        if beta >= 1.0 - 1e-8:
            # All vapor
            y = z.copy()
            x = z / np.maximum(K, EPSILON)
            x_sum = x.sum()
            if x_sum > EPSILON:
                x /= x_sum
            result.beta = 1.0
            result.x = x
            result.y = y
            result.K_values = K
            result.converged = True
            result.iterations = outer + 1
            result.message = "Single phase vapor"
            return result

        # Step 4: Fugacity coefficients
        # Liquid phase
        Z_liq = eos.Z_factor(P, T, phase='liquid',
                             mole_fractions=x.tolist(),
                             comp_params=comp_params, k_ij=k_ij)
        phi_liq = eos.fugacity_coefficient(P, T, Z_liq,
                                           mole_fractions=x.tolist(),
                                           comp_params=comp_params, k_ij=k_ij)

        # Vapor phase
        Z_vap = eos.Z_factor(P, T, phase='vapor',
                             mole_fractions=y.tolist(),
                             comp_params=comp_params, k_ij=k_ij)
        phi_vap = eos.fugacity_coefficient(P, T, Z_vap,
                                           mole_fractions=y.tolist(),
                                           comp_params=comp_params, k_ij=k_ij)

        # Step 5: Update K-values
        K_new = np.maximum(phi_liq, EPSILON) / np.maximum(phi_vap, EPSILON)

        # Check convergence
        error = np.max(np.abs(K_new / np.maximum(K, EPSILON) - 1.0))
        K = 0.7 * K_new + 0.3 * K  # Damping

        if error < tol_outer:
            converged = True
            result.beta = beta
            result.x = x
            result.y = y
            result.K_values = K
            result.converged = True
            result.iterations = outer + 1
            result.message = f"Converged in {outer + 1} iterations"
            break

    if not converged:
        result.beta = beta
        result.x = x
        result.y = y
        result.K_values = K
        result.converged = False
        result.iterations = outer + 1
        result.message = f"Not converged after {outer + 1} iterations"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Bubble Point & Dew Point
# ══════════════════════════════════════════════════════════════════════════════

def bubble_point_P(T: float, z: np.ndarray,
                    eos: Optional[CubicEoS] = None,
                    comp_params: Optional[List[EoSParameters]] = None,
                    k_ij: Optional[np.ndarray] = None,
                    P_guess: float = 1e5,
                    max_iter: int = 50,
                    tol: float = 1e-8) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """Bubble point pressure at given T.

    Finds pressure P where first vapor bubble forms (Σ y_i = 1, Σ K_i·x_i = 1).
    Since feed is liquid at bubble point, x_i = z_i.

    Algorithm:
      Outer loop: Update P so that Σ K_i·z_i = 1
      Inner: Fugacity equality → K-values → Σ K_i·z_i

    Args:
        T: Temperature [K].
        z: Feed (liquid) mole fractions.
        eos: EoS instance.
        comp_params: Component parameters.
        k_ij: Binary interaction matrix.
        P_guess: Initial pressure guess [Pa].
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (P_bubble, x_i, y_i, converged)
    """
    n = len(z)

    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    if comp_params is None and n == 1:
        comp_params = [eos._params]

    P = P_guess

    for it in range(max_iter):
        # Initialize K-values via Wilson
        K = np.ones(n)
        for i, cp in enumerate(comp_params):
            if cp.tc > 0 and cp.pc > 0:
                wf = 5.373 * (1.0 + cp.omega) * (1.0 - cp.tc / T)
                K[i] = (cp.pc / P) * math.exp(wf)
            K[i] = max(K[i], EPSILON)

        # Inner loop: refine K-values via fugacity equality
        x = z.copy()
        y = K * x
        y_sum = y.sum()
        if y_sum > EPSILON:
            y /= y_sum

        for _ in range(10):
            Z_liq = eos.Z_factor(P, T, phase='liquid',
                                 mole_fractions=x.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_liq = eos.fugacity_coefficient(P, T, Z_liq,
                                               mole_fractions=x.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            Z_vap = eos.Z_factor(P, T, phase='vapor',
                                 mole_fractions=y.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_vap = eos.fugacity_coefficient(P, T, Z_vap,
                                               mole_fractions=y.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            K_new = np.maximum(phi_liq, EPSILON) / np.maximum(phi_vap, EPSILON)

            if np.max(np.abs(K_new / np.maximum(K, EPSILON) - 1.0)) < tol:
                break

            K = 0.5 * K_new + 0.5 * K
            y = K * x
            y_sum = y.sum()
            if y_sum > EPSILON:
                y /= y_sum

        # Check: Σ K_i·z_i should be 1 at bubble point
        sum_Kz = np.sum(K * z)
        error = abs(sum_Kz - 1.0)

        if error < tol:
            x = z.copy()
            y = K * z
            y /= y.sum()
            return (P, x, y, True)

        # Update P: P_new = P_old · Σ K_i·z_i (Newton step)
        # d(sum_Kz)/dP approximation
        P = P * sum_Kz
        P = max(P, EPSILON)

    x = z.copy()
    y = K * z
    y_sum = y.sum()
    if y_sum > EPSILON:
        y /= y_sum
    return (P, x, y, False)


def bubble_point_T(P: float, z: np.ndarray,
                    eos: Optional[CubicEoS] = None,
                    comp_params: Optional[List[EoSParameters]] = None,
                    k_ij: Optional[np.ndarray] = None,
                    T_guess: float = 300.0,
                    max_iter: int = 50,
                    tol: float = 1e-8) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """Bubble point temperature at given P.

    Finds T where first vapor bubble forms at pressure P.

    Args:
        P: Pressure [Pa].
        z: Feed (liquid) mole fractions.
        eos: EoS instance.
        comp_params: Component parameters.
        k_ij: Binary interaction matrix.
        T_guess: Initial temperature guess [K].
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (T_bubble, x_i, y_i, converged)
    """
    n = len(z)

    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    if comp_params is None and n == 1:
        comp_params = [eos._params]

    T = T_guess

    for it in range(max_iter):
        K = np.ones(n)
        for i, cp in enumerate(comp_params):
            if cp.tc > 0:
                wf = 5.373 * (1.0 + cp.omega) * (1.0 - cp.tc / T)
                K[i] = (cp.pc / P) * math.exp(wf)
            K[i] = max(K[i], EPSILON)

        x = z.copy()
        y = K * x
        y_sum = y.sum()
        if y_sum > EPSILON:
            y /= y_sum

        for _ in range(10):
            Z_liq = eos.Z_factor(P, T, phase='liquid',
                                 mole_fractions=x.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_liq = eos.fugacity_coefficient(P, T, Z_liq,
                                               mole_fractions=x.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            Z_vap = eos.Z_factor(P, T, phase='vapor',
                                 mole_fractions=y.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_vap = eos.fugacity_coefficient(P, T, Z_vap,
                                               mole_fractions=y.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            K_new = np.maximum(phi_liq, EPSILON) / np.maximum(phi_vap, EPSILON)

            if np.max(np.abs(K_new / np.maximum(K, EPSILON) - 1.0)) < tol:
                break

            K = 0.5 * K_new + 0.5 * K
            y = K * x
            y_sum = y.sum()
            if y_sum > EPSILON:
                y /= y_sum

        sum_Kz = np.sum(K * z)
        error = abs(sum_Kz - 1.0)

        if error < tol:
            x = z.copy()
            y = K * z
            y /= y.sum()
            return (T, x, y, True)

        # Newton update: T_new = T * (1 + relaxation * (1 - 1/sum_Kz))
        T = T * (1.0 + 0.3 * math.log(max(sum_Kz, EPSILON)))
        T = max(T, 50.0)

    x = z.copy()
    y = K * z
    y_sum = y.sum()
    if y_sum > EPSILON:
        y /= y_sum
    return (T, x, y, False)


def dew_point_P(T: float, z: np.ndarray,
                 eos: Optional[CubicEoS] = None,
                 comp_params: Optional[List[EoSParameters]] = None,
                 k_ij: Optional[np.ndarray] = None,
                 P_guess: float = 1e5,
                 max_iter: int = 50,
                 tol: float = 1e-8) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """Dew point pressure at given T.

    Finds P where first liquid drop condenses (Σ x_i = 1, Σ z_i/K_i = 1).
    Since feed is vapor at dew point, y_i = z_i.

    Args:
        T: Temperature [K].
        z: Feed (vapor) mole fractions.
        eos: EoS instance.
        comp_params: Component parameters.
        k_ij: Binary interaction matrix.
        P_guess: Initial pressure guess [Pa].
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (P_dew, x_i, y_i, converged)
    """
    n = len(z)

    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    if comp_params is None and n == 1:
        comp_params = [eos._params]

    P = P_guess

    for it in range(max_iter):
        K = np.ones(n)
        for i, cp in enumerate(comp_params):
            if cp.tc > 0 and cp.pc > 0:
                wf = 5.373 * (1.0 + cp.omega) * (1.0 - cp.tc / T)
                K[i] = (cp.pc / P) * math.exp(wf)
            K[i] = max(K[i], EPSILON)

        y = z.copy()
        x = y / K
        x_sum = x.sum()
        if x_sum > EPSILON:
            x /= x_sum

        for _ in range(10):
            Z_liq = eos.Z_factor(P, T, phase='liquid',
                                 mole_fractions=x.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_liq = eos.fugacity_coefficient(P, T, Z_liq,
                                               mole_fractions=x.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            Z_vap = eos.Z_factor(P, T, phase='vapor',
                                 mole_fractions=y.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_vap = eos.fugacity_coefficient(P, T, Z_vap,
                                               mole_fractions=y.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            K_new = np.maximum(phi_liq, EPSILON) / np.maximum(phi_vap, EPSILON)

            if np.max(np.abs(K_new / np.maximum(K, EPSILON) - 1.0)) < tol:
                break

            K = 0.5 * K_new + 0.5 * K
            x = y / K
            x_sum = x.sum()
            if x_sum > EPSILON:
                x /= x_sum

        # At dew point: Σ z_i/K_i = 1
        sum_z_over_K = np.sum(z / np.maximum(K, EPSILON))
        error = abs(sum_z_over_K - 1.0)

        if error < tol:
            y = z.copy()
            x = z / K
            x /= x.sum()
            return (P, x, y, True)

        P = P / max(sum_z_over_K, EPSILON)
        P = max(P, EPSILON)

    y = z.copy()
    x = z / K
    x_sum = x.sum()
    if x_sum > EPSILON:
        x /= x_sum
    return (P, x, y, False)


def dew_point_T(P: float, z: np.ndarray,
                 eos: Optional[CubicEoS] = None,
                 comp_params: Optional[List[EoSParameters]] = None,
                 k_ij: Optional[np.ndarray] = None,
                 T_guess: float = 400.0,
                 max_iter: int = 50,
                 tol: float = 1e-8) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """Dew point temperature at given P.

    Args:
        P: Pressure [Pa].
        z: Feed (vapor) mole fractions.
        eos: EoS instance.
        comp_params: Component parameters.
        k_ij: Binary interaction matrix.
        T_guess: Initial temperature guess [K].
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (T_dew, x_i, y_i, converged)
    """
    n = len(z)

    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    if comp_params is None and n == 1:
        comp_params = [eos._params]

    T = T_guess

    for it in range(max_iter):
        K = np.ones(n)
        for i, cp in enumerate(comp_params):
            if cp.tc > 0:
                wf = 5.373 * (1.0 + cp.omega) * (1.0 - cp.tc / T)
                K[i] = (cp.pc / P) * math.exp(wf)
            K[i] = max(K[i], EPSILON)

        y = z.copy()
        x = y / K
        x_sum = x.sum()
        if x_sum > EPSILON:
            x /= x_sum

        for _ in range(10):
            Z_liq = eos.Z_factor(P, T, phase='liquid',
                                 mole_fractions=x.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_liq = eos.fugacity_coefficient(P, T, Z_liq,
                                               mole_fractions=x.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            Z_vap = eos.Z_factor(P, T, phase='vapor',
                                 mole_fractions=y.tolist(),
                                 comp_params=comp_params, k_ij=k_ij)
            phi_vap = eos.fugacity_coefficient(P, T, Z_vap,
                                               mole_fractions=y.tolist(),
                                               comp_params=comp_params, k_ij=k_ij)

            K_new = np.maximum(phi_liq, EPSILON) / np.maximum(phi_vap, EPSILON)

            if np.max(np.abs(K_new / np.maximum(K, EPSILON) - 1.0)) < tol:
                break

            K = 0.5 * K_new + 0.5 * K
            x = y / K
            x_sum = x.sum()
            if x_sum > EPSILON:
                x /= x_sum

        sum_z_over_K = np.sum(z / np.maximum(K, EPSILON))
        error = abs(sum_z_over_K - 1.0)

        if error < tol:
            y = z.copy()
            x = z / K
            x /= x.sum()
            return (T, x, y, True)

        T = T * (1.0 - 0.3 * math.log(max(sum_z_over_K, EPSILON)))
        T = max(T, 50.0)

    y = z.copy()
    x = z / K
    x_sum = x.sum()
    if x_sum > EPSILON:
        x /= x_sum
    return (T, x, y, False)


# ══════════════════════════════════════════════════════════════════════════════
# Phase Envelope Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_phase_envelope(z: np.ndarray,
                          comp_params: List[EoSParameters],
                          eos: Optional[CubicEoS] = None,
                          k_ij: Optional[np.ndarray] = None,
                          n_points: int = 50) -> Dict[str, np.ndarray]:
    """Build a P-T phase envelope for a mixture.

    Traces both the bubble point curve and dew point curve by
    stepping through a range of temperatures from low (liquid) to high (vapor).

    For single components, both curves are the same (vapor pressure curve).

    Args:
        z: Feed mole fractions.
        comp_params: Component EoSParameters.
        eos: EoS instance (PengRobinson default).
        k_ij: Binary interaction matrix.
        n_points: Number of points on each curve.

    Returns:
        Dict with:
          'T_bubble': bubble point temperatures [K]
          'P_bubble': bubble point pressures [Pa]
          'T_dew': dew point temperatures [K]
          'P_dew': dew point pressures [Pa]
          'T_crit': estimated critical temperature [K]
          'P_crit': estimated critical pressure [Pa]
    """
    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    n = len(z)

    if n == 1:
        # Pure component: vapor pressure curve
        cp = comp_params[0]
        t_range = np.linspace(cp.tc * 0.5, cp.tc * 0.99, n_points)
        p_curve = np.zeros(n_points)

        for i, T_val in enumerate(t_range):
            # For pure component, use Antoine / DIPPR vapor pressure
            # Simplified: use the vapor pressure from Pc and Tc
            Tr = T_val / cp.tc
            # Lee-Kesler correlation for vapor pressure
            omega = cp.omega
            f0 = (5.92714 - 6.09648 / Tr - 1.28862 * math.log(Tr) +
                  0.169347 * Tr ** 6)
            f1 = (15.2518 - 15.6875 / Tr - 13.4721 * math.log(Tr) +
                  0.43577 * Tr ** 6)
            ln_pr = f0 + omega * f1
            p_curve[i] = cp.pc * math.exp(ln_pr)

        # Critical point is the top
        return {
            'T_bubble': t_range.tolist(),
            'P_bubble': p_curve.tolist(),
            'T_dew': t_range.tolist(),
            'P_dew': p_curve.tolist(),
            'T_crit': float(cp.tc),
            'P_crit': float(cp.pc),
        }

    # Mixture: compute pseudo-critical point and trace curves
    # Estimate pseudo-critical via Kay's rule
    Tc_pseudo = sum(x_i * cp.tc for x_i, cp in zip(z, comp_params))
    Pc_pseudo = sum(x_i * cp.pc for x_i, cp in zip(z, comp_params))

    # Temperature range: from well below bubble point to near critical
    T_min = 0.4 * Tc_pseudo
    T_max = 0.98 * Tc_pseudo
    T_range = np.linspace(T_min, T_max, n_points)

    P_bubble = []
    T_bubble = []
    P_dew = []
    T_dew = []

    for T_val in T_range:
        # Bubble point
        try:
            Pb, _, _, conv_b = bubble_point_P(T_val, z, eos, comp_params, k_ij)
            if conv_b and Pb > 0 and Pb < Pc_pseudo * 2:
                P_bubble.append(Pb)
                T_bubble.append(T_val)
        except Exception:
            pass

        # Dew point
        try:
            Pd, _, _, conv_d = dew_point_P(T_val, z, eos, comp_params, k_ij)
            if conv_d and Pd > 0 and Pd < Pc_pseudo * 2:
                P_dew.append(Pd)
                T_dew.append(T_val)
        except Exception:
            pass

    return {
        'T_bubble': T_bubble,
        'P_bubble': P_bubble,
        'T_dew': T_dew,
        'P_dew': P_dew,
        'T_crit': float(Tc_pseudo),
        'P_crit': float(Pc_pseudo),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Critical Point Estimate (Heidemann-Khalil)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_critical_point(z: np.ndarray,
                             comp_params: List[EoSParameters],
                             eos: Optional[CubicEoS] = None,
                             k_ij: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Estimate the mixture critical temperature and pressure.

    Uses a simplified approach: the critical point is where the
    bubble and dew curves meet.

    For mixtures, this is a crude estimate. For rigorous critical
    point calculation, a full criticality condition solver
    (Heidemann-Khalil) would be needed.

    Args:
        z: Mole fractions.
        comp_params: Component parameters.
        eos: EoS instance.
        k_ij: Binary interaction matrix.

    Returns:
        (Tc_est, Pc_est) in [K] and [Pa].
    """
    if eos is None:
        eos = PengRobinson(tc=190.56, pc=4.599e6, omega=0.008, mw=16.043)

    # Simple Kay's rule weighting
    Tc = sum(x_i * cp.tc for x_i, cp in zip(z, comp_params))
    Pc = sum(x_i * cp.pc for x_i, cp in zip(z, comp_params))

    # Could refine by finding where bubble and dew curves converge
    # but for now, Kay's rule is a reasonable approximation
    return (Tc, Pc)
