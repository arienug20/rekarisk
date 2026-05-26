"""
Rekarisk — Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import pytest

from rekarisk.core.substance import Substance
from rekarisk.meteorology.meteorology import MeteorologicalState


# ---------------------------------------------------------------------------
# Substance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_substance() -> Substance:
    """Methane substance for EoS and source-term tests."""
    return Substance(
        name="Methane",
        formula="CH4",
        mw=16.043,
        tc=190.564,
        pc=4.599e6,
        omega=0.011,
        t_boil=111.63,
        hc=55.5e6,
    )


@pytest.fixture(scope="session")
def ethane_substance() -> Substance:
    """Ethane for mixture tests."""
    return Substance(
        name="Ethane",
        formula="C2H6",
        mw=30.07,
        tc=305.32,
        pc=4.872e6,
        omega=0.099,
        t_boil=184.55,
        hc=51.9e6,
    )


@pytest.fixture(scope="session")
def propane_substance() -> Substance:
    """Propane for fire/explosion tests."""
    return Substance(
        name="Propane",
        formula="C3H8",
        mw=44.10,
        tc=369.83,
        pc=4.248e6,
        omega=0.152,
        t_boil=231.11,
        hc=46.3e6,
    )


@pytest.fixture(scope="session")
def chlorine_substance() -> Substance:
    """Chlorine for toxic dispersion tests."""
    return Substance(
        name="Chlorine",
        formula="Cl2",
        mw=70.906,
        tc=416.9,
        pc=7.991e6,
        omega=0.073,
        t_boil=239.11,
    )


# ---------------------------------------------------------------------------
# Weather fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weather() -> MeteorologicalState:
    """Typical daytime D-stability weather conditions."""
    return MeteorologicalState(
        wind_speed_ms=5.0,
        wind_direction_deg=270.0,
        stability_class="D",
        ambient_temperature_k=293.15,
        ambient_pressure_pa=101325.0,
        relative_humidity=0.60,
        solar_radiation_wm2=400.0,
        cloud_cover_oktas=4,
    )


# ---------------------------------------------------------------------------
# CoolProp availability marker
# ---------------------------------------------------------------------------

try:
    import CoolProp  # noqa: F401

    HAVE_COOLPROP = True
except ImportError:
    HAVE_COOLPROP = False

skip_if_no_coolprop = pytest.mark.skipif(
    not HAVE_COOLPROP, reason="CoolProp not available"
)
