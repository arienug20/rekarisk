"""
Rekarisk — Cubic Equations of State.

Implements Peng-Robinson (PR) and Soave-Redlich-Kwong (SRK) cubic EoS
for pure components and mixtures.

For mixtures, van der Waals one-fluid mixing rules with
binary interaction parameters k_ij are used.

References:
  - Peng, D.-Y. & Robinson, D.B. (1976). IEC Fund. 15(1), 59–64.
  - Soave, G. (1972). Chem. Eng. Sci. 27(6), 1197–1203.
  - Reid, Prausnitz & Poling, "The Properties of Gases & Liquids", 4th ed.

All calculations in SI units: T [K], P [Pa], v [m³/mol], Z [-].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .constants import R, EPSILON


# ══════════════════════════════════════════════════════════════════════════════
# Cubic Solver Utilities
# ══════════════════════════════════════════════════════════════════════════════

def solve_cubic(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Solve the cubic equation: Z³ + a·Z² + b·Z + c = 0.

    Uses the analytical Cardano method with numerical fallback.
    Returns three real roots sorted ascending: (Z_min, Z_mid, Z_max).
    For EoS: Z_min = liquid-like, Z_max = vapor-like.

    Args:
        a: Coefficient of Z² term.
        b: Coefficient of Z term.
        c: Constant term.

    Returns:
        Tuple of three real roots (Z1 ≤ Z2 ≤ Z3).
    """
    # Depressed cubic: t³ + p·t + q = 0  where t = Z + a/3
    p = b - (a * a) / 3.0
    q = (2.0 * a ** 3) / 27.0 - (a * b) / 3.0 + c

    # Discriminant
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3

    if delta > 0:
        # One real root
        sqrt_delta = math.sqrt(delta)
        u = -q / 2.0 + sqrt_delta
        v = -q / 2.0 - sqrt_delta
        u_cbrt = math.copysign(abs(u) ** (1.0 / 3.0), u)
        v_cbrt = math.copysign(abs(v) ** (1.0 / 3.0), v)
        z1 = u_cbrt + v_cbrt - a / 3.0
        # Return same root repeated for all slots
        return (z1, z1, z1)
    else:
        # Three real roots (or repeated)
        phi = math.acos(-q / (2.0 * math.sqrt(-(p / 3.0) ** 3))) if p < -EPSILON else 0.0
        r = 2.0 * math.sqrt(-p / 3.0)
        z1 = r * math.cos(phi / 3.0) - a / 3.0
        z2 = r * math.cos((phi + 2.0 * math.pi) / 3.0) - a / 3.0
        z3 = r * math.cos((phi + 4.0 * math.pi) / 3.0) - a / 3.0
        roots = sorted([z1, z2, z3])
        return (roots[0], roots[1], roots[2])


def solve_cubic_newton(a: float, b: float, c: float,
                        initial_guess: float = 1.0,
                        max_iter: int = 100,
                        tol: float = 1e-10) -> float:
    """Solve cubic Z³ + aZ² + bZ + c = 0 using Newton-Raphson.

    Used as fallback when analytical solution is numerically unstable.

    Args:
        a, b, c: Cubic coefficients.
        initial_guess: Starting Z value.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        The converged root.
    """
    Z = initial_guess
    for _ in range(max_iter):
        f = Z ** 3 + a * Z * Z + b * Z + c
        df = 3.0 * Z * Z + 2.0 * a * Z + b
        if abs(df) < EPSILON:
            break
        dZ = f / df
        Z -= dZ
        if abs(dZ) < tol:
            break
    return Z


# ══════════════════════════════════════════════════════════════════════════════
# Common EoS base class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EoSParameters:
    """EoS parameters for a single component."""
    # Attraction parameter a [Pa·(m³/mol)²]
    a: float = 0.0
    # Co-volume parameter b [m³/mol]
    b: float = 0.0
    # Temperature-dependent alpha(T)
    alpha: float = 1.0
    # Critical temperature [K]
    tc: float = 300.0
    # Critical pressure [Pa]
    pc: float = 1e6
    # Acentric factor
    omega: float = 0.0
    # Molecular weight [g/mol]
    mw: float = 16.0


class CubicEoS:
    """Base class for cubic equations of state.

    Subclasses must define:
      - _compute_a(Tc, Pc): raw attraction parameter
      - _compute_b(Tc, Pc): co-volume parameter
      - _kappa(omega): alpha-function correlation parameter
      - _u, _w: EoS-specific constants for cubic form

    Generic cubic: P = RT/(v-b) - a·α(T) / [(v+u·b)(v+w·b)]
      PR: u = 1+√2 ≈ 2.414, w = 1-√2 ≈ -0.414
      SRK: u = 1, w = 0

    Cubic in Z:
      Z³ + a1·Z² + a2·Z + a3 = 0
    """

    # EoS-specific constants (overridden by subclasses)
    _u: float = 1.0
    _w: float = 0.0
    _name: str = "BaseEoS"

    def __init__(self, params: Optional[EoSParameters] = None, **kwargs):
        """Initialize EoS for a pure component.

        Args:
            params: Pre-computed EoSParameters.
            **kwargs: Alternatively, pass tc, pc, omega, mw directly.
        """
        if params is not None:
            self._params = params
        else:
            tc = kwargs.get('tc', 300.0)
            pc = kwargs.get('pc', 1e6)
            omega = kwargs.get('omega', 0.0)
            mw = kwargs.get('mw', 16.0)
            a = self._compute_a(tc, pc)
            b = self._compute_b(tc, pc)
            self._params = EoSParameters(
                a=a, b=b, alpha=1.0, tc=tc, pc=pc, omega=omega, mw=mw
            )

    # ── Subclass overrides ──

    @staticmethod
    def _compute_a(tc: float, pc: float) -> float:
        """Raw attraction parameter a [Pa·(m³/mol)²]."""
        raise NotImplementedError

    @staticmethod
    def _compute_b(tc: float, pc: float) -> float:
        """Co-volume parameter b [m³/mol]."""
        raise NotImplementedError

    @staticmethod
    def _kappa(omega: float) -> float:
        """Alpha-function parameter κ."""
        raise NotImplementedError

    # ── Alpha function ──

    def _alpha(self, T: float) -> float:
        """Temperature-dependent alpha(T) = [1 + κ·(1 - √Tr)]²."""
        Tr = T / self._params.tc
        if Tr <= 0:
            return 0.0
        k = self._kappa(self._params.omega)
        sqrt_tr = math.sqrt(Tr)
        alpha = (1.0 + k * (1.0 - sqrt_tr)) ** 2
        return max(alpha, EPSILON)

    # ── Mixing rule parameters ──

    def a_alpha_mix(self, T: float, mole_fractions: List[float],
                    comp_params: List[EoSParameters],
                    k_ij: Optional[np.ndarray] = None) -> float:
        """Mixture a·α(T) using van der Waals one-fluid mixing rule.

        aα_mix = Σ_i Σ_j x_i·x_j·√(a_i·α_i·a_j·α_j)·(1 - k_ij)

        Args:
            T: Temperature [K].
            mole_fractions: List of mole fractions x_i.
            comp_params: EoSParameters for each component.
            k_ij: Binary interaction matrix (N×N). Default: zeros.

        Returns:
            Mixture a·α [Pa·(m³/mol)²].
        """
        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        # Precompute a·α for each component
        a_alpha_i = []
        for i, params in enumerate(comp_params):
            ai = params.a
            bi = params.b
            # Recompute alpha at T
            Tr = T / params.tc
            if Tr > 0:
                k = self._kappa(params.omega)
                sqrt_tr = math.sqrt(Tr)
                alpha_i = max((1.0 + k * (1.0 - sqrt_tr)) ** 2, EPSILON)
            else:
                alpha_i = EPSILON
            a_alpha_i.append(ai * alpha_i)

        a_alpha = 0.0
        for i in range(n):
            for j in range(n):
                cross = math.sqrt(a_alpha_i[i] * a_alpha_i[j])
                a_alpha += mole_fractions[i] * mole_fractions[j] * cross * (1.0 - k_ij[i, j])

        return a_alpha

    def b_mix(self, mole_fractions: List[float],
              comp_params: List[EoSParameters]) -> float:
        """Mixture co-volume: b_mix = Σ_i x_i·b_i."""
        return sum(x_i * p.b for x_i, p in zip(mole_fractions, comp_params))

    # ── Cubic coefficients ──

    def _cubic_coefficients(self, T: float, P: float,
                             a_alpha: float, b: float) -> Tuple[float, float, float]:
        """Compute dimensionless cubic coefficients A, B for EoS.

        Z³ + a1·Z² + a2·Z + a3 = 0

        Returns (a1, a2, a3).
        """
        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        u = self._u
        w = self._w

        a1 = (u + w - 1.0) * B - 1.0
        a2 = A - (u + w - u * w) * B * B - u * B - w * B
        a3 = -(A * B + u * w * B * B * (1.0 + B))

        return (a1, a2, a3)

    # ── Z-factor ──

    def Z_factor(self, P: float, T: float,
                  phase: str = 'vapor',
                  mole_fractions: Optional[List[float]] = None,
                  comp_params: Optional[List[EoSParameters]] = None,
                  k_ij: Optional[np.ndarray] = None) -> float:
        """Compute compressibility factor Z = Pv/(RT).

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            phase: 'vapor' (largest root) or 'liquid' (smallest root).
            mole_fractions: Mole fractions (None for pure).
            comp_params: Component parameters (None for pure).
            k_ij: Binary interaction matrix.

        Returns:
            Compressibility factor Z [-].
        """
        if mole_fractions is not None and comp_params is not None:
            a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
            b = self.b_mix(mole_fractions, comp_params)
        else:
            a_alpha = self._params.a * self._alpha(T)
            b = self._params.b

        a1, a2, a3 = self._cubic_coefficients(T, P, a_alpha, b)

        try:
            roots = solve_cubic(a1, a2, a3)
            Z_min, Z_mid, Z_max = roots

            if phase == 'liquid':
                return max(Z_min, EPSILON)
            else:
                return max(Z_max, EPSILON)
        except (ValueError, OverflowError):
            # Newton fallback
            guess = 0.01 if phase == 'liquid' else 1.0
            Z = solve_cubic_newton(a1, a2, a3, initial_guess=guess)
            return max(Z, EPSILON)

    def Z_factors(self, P: float, T: float,
                   mole_fractions: Optional[List[float]] = None,
                   comp_params: Optional[List[EoSParameters]] = None,
                   k_ij: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Return both liquid and vapor Z factors.

        Returns (Z_liquid, Z_vapor).
        """
        if mole_fractions is not None and comp_params is not None:
            a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
            b = self.b_mix(mole_fractions, comp_params)
        else:
            a_alpha = self._params.a * self._alpha(T)
            b = self._params.b

        a1, a2, a3 = self._cubic_coefficients(T, P, a_alpha, b)

        try:
            roots = solve_cubic(a1, a2, a3)
            return (max(roots[0], EPSILON), max(roots[2], EPSILON))
        except (ValueError, OverflowError):
            Z_liq = solve_cubic_newton(a1, a2, a3, initial_guess=0.01)
            Z_vap = solve_cubic_newton(a1, a2, a3, initial_guess=1.0)
            return (max(Z_liq, EPSILON), max(Z_vap, EPSILON))

    # ── Density ──

    def density(self, P: float, T: float,
                phase: str = 'vapor',
                mole_fractions: Optional[List[float]] = None,
                comp_params: Optional[List[EoSParameters]] = None,
                k_ij: Optional[np.ndarray] = None) -> float:
        """Compute density from Z-factor.

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            phase: 'vapor' or 'liquid'.
            mole_fractions: For mixtures.
            comp_params: Component parameters for mixtures.
            k_ij: Binary interaction matrix.

        Returns:
            Density [kg/m³].
        """
        Z = self.Z_factor(P, T, phase=phase,
                          mole_fractions=mole_fractions,
                          comp_params=comp_params, k_ij=k_ij)

        # Average molecular weight for mixtures
        if mole_fractions is not None and comp_params is not None:
            mw_avg = sum(x_i * p.mw for x_i, p in zip(mole_fractions, comp_params))
        else:
            mw_avg = self._params.mw

        mw_kg_per_mol = mw_avg * 0.001  # g/mol → kg/mol
        return (P * mw_kg_per_mol) / (Z * R * T)

    # ── Fugacity coefficient ──

    def fugacity_coefficient(self, P: float, T: float, Z: float,
                              mole_fractions: Optional[List[float]] = None,
                              comp_params: Optional[List[EoSParameters]] = None,
                              k_ij: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute fugacity coefficients φ_i for each component in a mixture.

        Based on:
          ln(φ_i) = b_i/b·(Z - 1) - ln(Z - B)
                    + A/(B·√(u² - 4w)) · (b_i/b - δ_i) · ln((Z + β·B)/(Z + σ·B))

        where:
          δ_i = 2/√(aα) · Σ_j x_j·√(a_j·α_j)·(1 - k_ij)

        For pure component, returns single element array.

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            Z: Compressibility factor (from Z_factor).
            mole_fractions: Mole fractions (None for pure, assumed [1.0]).
            comp_params: Component parameters.
            k_ij: Binary interaction matrix.

        Returns:
            Array of fugacity coefficients φ_i.
        """
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        # Get mixture parameters at T
        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        u = self._u
        w = self._w

        # Compute sqrt(a_i·α_i) for each component
        ai_alpha_sqrt = []
        bi = []
        for i, params in enumerate(comp_params):
            Tr = T / params.tc
            if Tr > 0:
                k = self._kappa(params.omega)
                alpha_i = max((1.0 + k * (1.0 - math.sqrt(Tr))) ** 2, EPSILON)
            else:
                alpha_i = EPSILON
            ai_alpha_sqrt.append(math.sqrt(params.a * alpha_i))
            bi.append(params.b)

        # Compute delta_i for each component
        delta_i = np.zeros(n)
        sqrt_a_alpha = math.sqrt(a_alpha)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += mole_fractions[j] * ai_alpha_sqrt[j] * (1.0 - k_ij[i, j])
            delta_i[i] = 2.0 * s / sqrt_a_alpha if sqrt_a_alpha > EPSILON else 0.0

        # Compute fugacity coefficients
        phi = np.ones(n)

        # Ln(Z - B) term
        Z_minus_B = Z - B
        if Z_minus_B <= EPSILON:
            Z_minus_B = EPSILON

        # Determine mixing rule-specific terms
        if abs(u - w) < EPSILON:
            # Special case: u = w (not typical for PR or SRK)
            ln_term = -B / Z_minus_B
        else:
            # Generic: Z³ + ... → ln((Z + B*u)/(Z + B*w))
            # For PR: u = 1+√2, w = 1-√2
            # For SRK: u = 1, w = 0
            denominator = -B * (u - w)
            if abs(denominator) < EPSILON:
                phi[:] = 1.0
                return phi

            if Z + u * B <= EPSILON or Z + w * B <= EPSILON:
                ln_term = 0.0
            else:
                ln_term = math.log((Z + u * B) / (Z + w * B))

            factor = A / (B * (u - w)) if abs(B * (u - w)) > EPSILON else 0.0

            for i in range(n):
                phi_i = (bi[i] / b) * (Z - 1.0) - math.log(Z_minus_B)
                phi_i += factor * ((bi[i] / b) - delta_i[i]) * ln_term
                phi[i] = math.exp(phi_i)

        return phi

    # ── Departure functions ──

    def enthalpy_departure(self, P: float, T: float, Z: float,
                            mole_fractions: Optional[List[float]] = None,
                            comp_params: Optional[List[EoSParameters]] = None,
                            k_ij: Optional[np.ndarray] = None) -> float:
        """Enthalpy departure (H - H_ideal) [J/mol].

        H_dep = R·T·(Z - 1) + [T·(daα/dT) - aα] / (b·(u - w)) · ln((Z + uB)/(Z + wB))

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            Z: Compressibility factor.
            mole_fractions: Mole fractions for mixture.
            comp_params: Component parameters.
            k_ij: Binary interaction matrix.

        Returns:
            Enthalpy departure [J/mol].
        """
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        # Mixture parameters
        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        u = self._u
        w = self._w

        # Compute d(aα)/dT for the mixture using numerical derivative
        # aα(T) = ΣΣ x_i·x_j·(1-k_ij)·√(a_i·α_i(T)·a_j·α_j(T))
        # d(aα)/dT = ΣΣ x_i·x_j·(1-k_ij)·0.5·(a_i·a_j/√(a_i·α_i·a_j·α_j))·(α_i·dα_j/dT + α_j·dα_i/dT)
        dT = max(1e-4 * T, 1e-3)
        a_alpha_plus = self.a_alpha_mix(T + dT, mole_fractions, comp_params, k_ij)
        a_alpha_minus = self.a_alpha_mix(T - dT, mole_fractions, comp_params, k_ij)
        da_alpha_dT = (a_alpha_plus - a_alpha_minus) / (2.0 * dT)

        if abs(u - w) < EPSILON:
            return R * T * (Z - 1.0)

        if Z + u * B <= EPSILON or Z + w * B <= EPSILON:
            return R * T * (Z - 1.0)

        ln_term = math.log((Z + u * B) / (Z + w * B))
        factor = (T * da_alpha_dT - a_alpha) / (b * (u - w)) if abs(b * (u - w)) > EPSILON else 0.0

        return R * T * (Z - 1.0) + factor * ln_term

    def entropy_departure(self, P: float, T: float, Z: float,
                           mole_fractions: Optional[List[float]] = None,
                           comp_params: Optional[List[EoSParameters]] = None,
                           k_ij: Optional[np.ndarray] = None) -> float:
        """Entropy departure (S - S_ideal) [J/(mol·K)].

        S_dep = R·ln(Z - B) + [daα/dT] / (b·(u - w)) · ln((Z + uB)/(Z + wB))

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            Z: Compressibility factor.
            mole_fractions: Mole fractions for mixture.
            comp_params: Component parameters.
            k_ij: Binary interaction matrix.

        Returns:
            Entropy departure [J/(mol·K)].
        """
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        B = b * P / (R * T)

        # Numerical derivative
        dT = max(1e-4 * T, 1e-3)
        a_alpha_plus = self.a_alpha_mix(T + dT, mole_fractions, comp_params, k_ij)
        a_alpha_minus = self.a_alpha_mix(T - dT, mole_fractions, comp_params, k_ij)
        da_alpha_dT = (a_alpha_plus - a_alpha_minus) / (2.0 * dT)

        u = self._u
        w = self._w

        Z_minus_B = max(Z - B, EPSILON)

        if abs(u - w) < EPSILON:
            return R * math.log(Z_minus_B)

        if Z + u * B <= EPSILON or Z + w * B <= EPSILON:
            return R * math.log(Z_minus_B)

        ln_term = math.log((Z + u * B) / (Z + w * B))
        factor = da_alpha_dT / (b * (u - w)) if abs(b * (u - w)) > EPSILON else 0.0

        return R * math.log(Z_minus_B) + factor * ln_term

    def cp_departure(self, P: float, T: float, Z: float,
                      mole_fractions: Optional[List[float]] = None,
                      comp_params: Optional[List[EoSParameters]] = None,
                      k_ij: Optional[np.ndarray] = None) -> float:
        """Heat capacity departure (Cp - Cp_ideal) [J/(mol·K)].

        Uses second derivative of aα.

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            Z: Compressibility factor.
            mole_fractions: Mole fractions for mixture.
            comp_params: Component parameters.
            k_ij: Binary interaction matrix.

        Returns:
            Cp departure [J/(mol·K)].
        """
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        # Numerical derivatives
        dT = max(1e-4 * T, 1e-3)
        a_alpha_plus = self.a_alpha_mix(T + dT, mole_fractions, comp_params, k_ij)
        a_alpha_minus = self.a_alpha_mix(T - dT, mole_fractions, comp_params, k_ij)
        da_alpha_dT = (a_alpha_plus - a_alpha_minus) / (2.0 * dT)

        # Second derivative
        a_alpha_2plus = self.a_alpha_mix(T + 2 * dT, mole_fractions, comp_params, k_ij)
        d2a_alpha_dT2 = (a_alpha_2plus - 2.0 * a_alpha + a_alpha_minus) / (dT * dT)

        # dZ/dT derivative
        dT_small = max(1e-4 * T, 1e-2)
        Z_plus = self.Z_factor(P, T + dT_small, phase='vapor',
                               mole_fractions=mole_fractions,
                               comp_params=comp_params, k_ij=k_ij)
        Z_minus = self.Z_factor(P, T - dT_small, phase='vapor',
                                mole_fractions=mole_fractions,
                                comp_params=comp_params, k_ij=k_ij)
        dZ_dT = (Z_plus - Z_minus) / (2.0 * dT_small)

        u = self._w  # Note: using _w as u for PR-style
        w = self._w

        # Recalculate u, w for the specific EoS
        u_eos = self._u
        w_eos = self._w

        if abs(u_eos - w_eos) < EPSILON:
            return -R * (1.0 + T * dZ_dT) / max(Z - 1.0, EPSILON)

        Z_plus_uB = Z + u_eos * B
        Z_plus_wB = Z + w_eos * B
        if Z_plus_uB <= EPSILON or Z_plus_wB <= EPSILON:
            return 0.0

        ln_term = math.log(Z_plus_uB / Z_plus_wB)
        factor = T * d2a_alpha_dT2 / (b * (u_eos - w_eos)) if abs(b * (u_eos - w_eos)) > EPSILON else 0.0

        cp_dep = factor * ln_term - R
        return cp_dep

    def speed_of_sound(self, P: float, T: float, Z: float,
                        cp_cv_ratio: float = 1.3,
                        mole_fractions: Optional[List[float]] = None,
                        comp_params: Optional[List[EoSParameters]] = None) -> float:
        """Speed of sound [m/s].

        c = √(γ·R·T·Z / MW_kg)   (ideal gas approximation improved with Z)

        More rigorously:
          c² = (Cp/Cv) · (∂P/∂ρ)_T · (1/ρ²)

        Uses simplified form with Z correction.

        Args:
            P: Pressure [Pa].
            T: Temperature [K].
            Z: Compressibility factor.
            cp_cv_ratio: Cp/Cv ratio γ.
            mole_fractions: For mixtures.
            comp_params: Component parameters.

        Returns:
            Speed of sound [m/s].
        """
        if mole_fractions is not None and comp_params is not None:
            mw_avg = sum(x_i * p.mw for x_i, p in zip(mole_fractions, comp_params))
        else:
            mw_avg = self._params.mw

        mw_kg = mw_avg * 0.001  # g/mol → kg/mol

        # Basic: c = sqrt(γ·Z·R·T / MW_kg)
        # More realistic version accounts for (∂P/∂ρ) from EoS
        # Using simplified form: c = sqrt(γ·Z·R·T/MW_kg)
        c_squared = cp_cv_ratio * Z * R * T / mw_kg

        return math.sqrt(max(c_squared, EPSILON))


# ══════════════════════════════════════════════════════════════════════════════
# Peng-Robinson EoS
# ══════════════════════════════════════════════════════════════════════════════

class PengRobinson(CubicEoS):
    """Peng-Robinson Equation of State (1976).

    P = RT/(v - b) - a·α(T) / (v² + 2b·v - b²)

    Cubic in Z:
      Z³ - (1 - B)·Z² + (A - 2B - 3B²)·Z - (AB - B² - B³) = 0

    where:
      A = a·α·P/(RT)²
      B = b·P/(RT)

    Parameters:
      a = 0.45724·R²·Tc²/Pc
      b = 0.07780·R·Tc/Pc
      κ = 0.37464 + 1.54226·ω - 0.26992·ω²
      α(T) = [1 + κ·(1 - √Tr)]²
    """

    _u: float = 1.0 + math.sqrt(2.0)  # ≈ 2.414
    _w: float = 1.0 - math.sqrt(2.0)  # ≈ -0.414
    _name: str = "Peng-Robinson"

    @staticmethod
    def _compute_a(tc: float, pc: float) -> float:
        return 0.45724 * R * R * tc * tc / pc

    @staticmethod
    def _compute_b(tc: float, pc: float) -> float:
        return 0.07780 * R * tc / pc

    @staticmethod
    def _kappa(omega: float) -> float:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega * omega

    # Override cubic coefficients for PR-specific form (more stable numerically)
    def _cubic_coefficients(self, T: float, P: float,
                             a_alpha: float, b: float) -> Tuple[float, float, float]:
        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        # PR standard: Z³ - (1-B)·Z² + (A-2B-3B²)·Z - (AB-B²-B³) = 0
        # → a1·Z³ + a2·Z² + a3·Z + a4 = 0 where a1=1
        a1 = -(1.0 - B)
        a2 = A - 2.0 * B - 3.0 * B * B
        a3 = -(A * B - B * B - B * B * B)

        return (a1, a2, a3)

    def fugacity_coefficient(self, P: float, T: float, Z: float,
                              mole_fractions: Optional[List[float]] = None,
                              comp_params: Optional[List[EoSParameters]] = None,
                              k_ij: Optional[np.ndarray] = None) -> np.ndarray:
        """PR-specific analytical fugacity coefficient."""
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        # Component-specific parameters
        ai_alpha_sqrt = []
        bi = []
        for i, params in enumerate(comp_params):
            Tr = T / params.tc
            if Tr > 0:
                k = self._kappa(params.omega)
                alpha_i = max((1.0 + k * (1.0 - math.sqrt(Tr))) ** 2, EPSILON)
            else:
                alpha_i = EPSILON
            ai_alpha_sqrt.append(math.sqrt(params.a * alpha_i))
            bi.append(params.b)

        sqrt_a_alpha = math.sqrt(a_alpha)
        delta_i = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += mole_fractions[j] * ai_alpha_sqrt[j] * (1.0 - k_ij[i, j])
            delta_i[i] = 2.0 * s / sqrt_a_alpha if sqrt_a_alpha > EPSILON else 0.0

        phi = np.ones(n)
        Z_minus_B = max(Z - B, EPSILON)

        denominator = self._u - self._w
        if abs(denominator) < EPSILON:
            return phi

        if Z + self._u * B <= EPSILON or Z + self._w * B <= EPSILON:
            return phi

        ln_term = math.log((Z + self._u * B) / (Z + self._w * B))
        factor = A / (B * denominator) if abs(B * denominator) > EPSILON else 0.0

        for i in range(n):
            phi_i = (bi[i] / b) * (Z - 1.0) - math.log(Z_minus_B)
            phi_i += factor * ((bi[i] / b) - delta_i[i]) * ln_term
            phi[i] = math.exp(min(phi_i, 50.0))  # Prevent overflow

        return phi


# ══════════════════════════════════════════════════════════════════════════════
# Soave-Redlich-Kwong (SRK) EoS
# ══════════════════════════════════════════════════════════════════════════════

class SoaveRedlichKwong(CubicEoS):
    """Soave-Redlich-Kwong Equation of State (1972).

    P = RT/(v - b) - a·α(T) / (v·(v + b))

    Cubic in Z:
      Z³ - Z² + (A - B - B²)·Z - AB = 0

    where:
      A = a·α·P/(RT)²
      B = b·P/(RT)

    Parameters:
      a = 0.42747·R²·Tc²/Pc
      b = 0.08664·R·Tc/Pc
      κ = 0.480 + 1.574·ω - 0.176·ω²
      α(T) = [1 + κ·(1 - √Tr)]²
    """

    _u: float = 1.0
    _w: float = 0.0
    _name: str = "Soave-Redlich-Kwong"

    @staticmethod
    def _compute_a(tc: float, pc: float) -> float:
        return 0.42747 * R * R * tc * tc / pc

    @staticmethod
    def _compute_b(tc: float, pc: float) -> float:
        return 0.08664 * R * tc / pc

    @staticmethod
    def _kappa(omega: float) -> float:
        return 0.480 + 1.574 * omega - 0.176 * omega * omega

    # Override cubic coefficients for SRK-specific form
    def _cubic_coefficients(self, T: float, P: float,
                             a_alpha: float, b: float) -> Tuple[float, float, float]:
        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        # SRK: Z³ - Z² + (A - B - B²)·Z - AB = 0
        # → a1·Z³ + a2·Z² + a3·Z + a4 = 0
        a1 = -1.0
        a2 = A - B - B * B
        a3 = -A * B

        return (a1, a2, a3)

    def fugacity_coefficient(self, P: float, T: float, Z: float,
                              mole_fractions: Optional[List[float]] = None,
                              comp_params: Optional[List[EoSParameters]] = None,
                              k_ij: Optional[np.ndarray] = None) -> np.ndarray:
        """SRK-specific analytical fugacity coefficient."""
        if mole_fractions is None:
            mole_fractions = [1.0]
        if comp_params is None:
            comp_params = [self._params]

        n = len(mole_fractions)
        if k_ij is None:
            k_ij = np.zeros((n, n))

        a_alpha = self.a_alpha_mix(T, mole_fractions, comp_params, k_ij)
        b = self.b_mix(mole_fractions, comp_params)

        A = a_alpha * P / (R * R * T * T)
        B = b * P / (R * T)

        # Component-specific parameters
        ai_alpha_sqrt = []
        bi = []
        for i, params in enumerate(comp_params):
            Tr = T / params.tc
            if Tr > 0:
                k = self._kappa(params.omega)
                alpha_i = max((1.0 + k * (1.0 - math.sqrt(Tr))) ** 2, EPSILON)
            else:
                alpha_i = EPSILON
            ai_alpha_sqrt.append(math.sqrt(params.a * alpha_i))
            bi.append(params.b)

        sqrt_a_alpha = math.sqrt(a_alpha)
        delta_i = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += mole_fractions[j] * ai_alpha_sqrt[j] * (1.0 - k_ij[i, j])
            delta_i[i] = 2.0 * s / sqrt_a_alpha if sqrt_a_alpha > EPSILON else 0.0

        phi = np.ones(n)
        Z_minus_B = max(Z - B, EPSILON)

        # SRK: ln((Z + B)/Z) term (since u=1, w=0)
        if Z <= 0:
            return phi

        ln_term = math.log((Z + B) / Z)
        factor = A / B if abs(B) > EPSILON else 0.0

        for i in range(n):
            phi_i = (bi[i] / b) * (Z - 1.0) - math.log(Z_minus_B)
            phi_i += factor * ((bi[i] / b) - delta_i[i]) * ln_term
            phi[i] = math.exp(min(phi_i, 50.0))

        return phi


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Factory Function
# ══════════════════════════════════════════════════════════════════════════════

def create_eos(eos_type: str = 'pr',
               tc: float = 300.0,
               pc: float = 1e6,
               omega: float = 0.0,
               mw: float = 16.0) -> CubicEoS:
    """Create an EoS instance for a pure component.

    Args:
        eos_type: 'pr' for Peng-Robinson, 'srk' for Soave-Redlich-Kwong.
        tc: Critical temperature [K].
        pc: Critical pressure [Pa].
        omega: Acentric factor [-].
        mw: Molecular weight [g/mol].

    Returns:
        CubicEoS instance.
    """
    if eos_type.lower() == 'pr':
        return PengRobinson(tc=tc, pc=pc, omega=omega, mw=mw)
    elif eos_type.lower() == 'srk':
        return SoaveRedlichKwong(tc=tc, pc=pc, omega=omega, mw=mw)
    else:
        raise ValueError(f"Unknown EoS type: '{eos_type}'. Use 'pr' or 'srk'.")


def create_mixture_eos(eos_type: str = 'pr',
                        substances: Optional[List] = None,
                        mole_fractions: Optional[List[float]] = None,
                        k_ij: Optional[np.ndarray] = None) -> Tuple[CubicEoS, List[EoSParameters], List[float]]:
    """Create an EoS instance configured for a mixture.

    Args:
        eos_type: 'pr' or 'srk'.
        substances: List of Substance objects with tc, pc, omega, mw attributes.
        mole_fractions: Mole fractions list.
        k_ij: Binary interaction matrix.

    Returns:
        Tuple of (CubicEoS instance, component_params_list, mole_fractions_list).
    """
    if eos_type.lower() == 'pr':
        eos = PengRobinson.__new__(PengRobinson)
    else:
        eos = SoaveRedlichKwong.__new__(SoaveRedlichKwong)

    comp_params = []
    for sub in substances:
        cp = EoSParameters(
            a=eos._compute_a(sub.critical_temperature, sub.critical_pressure),
            b=eos._compute_b(sub.critical_temperature, sub.critical_pressure),
            alpha=1.0,
            tc=sub.critical_temperature,
            pc=sub.critical_pressure,
            omega=sub.acentric_factor or 0.0,
            mw=sub.molecular_weight
        )
        comp_params.append(cp)

    return eos, comp_params, mole_fractions or []
