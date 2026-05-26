# Rekarisk Validation & Benchmarking

This document describes the validation test suite for Rekarisk models,
including test case descriptions, expected versus actual behavior,
known limitations, and comparison with published benchmarks.

---

## 1. Test Suite Overview

The validation test suite is located in `tests/` and organized by model category:

| File | Lines | Test Classes | Models Covered |
|------|-------|------------|----------------|
| `test_source_term.py` | ~400 | 5 | Orifice, PSV, Vessel Blowdown, Pool Evaporation |
| `test_dispersion.py` | ~400 | 3 | Gaussian Plume, Gaussian Puff, Dense Gas |
| `test_fire.py` | ~400 | 3 | Pool Fire, Jet Fire, BLEVE |
| `test_explosion.py` | ~400 | 3 | TNT Equivalency, TNO Multi-Energy, BST |
| `test_vulnerability.py` | ~300 | 4 | Probit Core, Thermal, Overpressure, Toxic |
| `test_qra.py` | ~300 | 5 | Event Tree, Failure Frequency, Ignition, IR, FN, Risk Matrix |
| `test_eos.py` | ~200 | 3 | Peng-Robinson, PR Mixture, SRK |
| `test_meteorology.py` | ~200 | 3 | Stability, Sigma, Wind Profile, Air Density |
| **Total** | **~2,600** | **29** | — |

---

## 2. Source Term Test Cases

### 2.1 Liquid Orifice Discharge

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_water_10mm_hole_5bar` | Liquid water through 10mm hole, 5 bar gauge | mdot ~1.5 kg/s (order of magnitude check) | Physical benchmark |
| `test_liquid_no_pressure_difference` | Zero ΔP → zero flow | mdot = 0 | Sanity check |
| `test_liquid_with_head` | 5m head → higher mdot than no head | mdot(head) > mdot(no head) | Physical check |
| `test_liquid_orifice_area` | Verify area = πd²/4 | Computed vs analytical | Mathematical |

### 2.2 Gas Orifice Discharge

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_gas_choked_flow` | 50 bar → atm, k=1.3 | is_choked = True | Physical check |
| `test_gas_subsonic_flow` | 1.1 bar → 1.0 atm | is_choked = False | Physical check |
| `test_gas_hydrogen_vs_methane` | H₂ velocity > CH₄ velocity | v(H₂) > v(CH₄) | Physics (lighter gas) |
| `test_choked_pressure_calculation` | Critical ratio for k=1.4 | P_choked/P_up ≈ 0.528 | Physical benchmark |

### 2.3 PSV (API 520)

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_gas_liquid_steam_area` | All three fluids | A_required > 0 | Sanity check |
| `test_orifice_designation_selection` | Known API areas | D for small, K for 1186, T+ for largest | Specification |

### 2.4 Vessel Blowdown

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_pressure_decreases` | P(t) decreases monotonically | P_final < P_initial | Physical check |
| `test_temperature_decreases` | Adiabatic expansion | T_final < T_initial | Physical check |
| `test_mass_decreases` | Mass leaves vessel | m_final < m_initial | Conservation |
| `test_total_mass_released_positive` | Total mass released > 0 | m_released > 0 | Sanity check |

### 2.5 Pool Evaporation

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_evaporation_rate_positive` | Gasoline on concrete | evap rate > 0 | Sanity check |
| `test_high_wind_increases_evaporation` | 10 m/s vs 1 m/s | Higher wind → higher rate | Physical check |
| `test_bunded_pool_area_limited` | Bunded area = 50 m² | Pool area ≤ 50 m² | Physical check |
| `test_mass_conservation` | Mass remaining + evaporated | ≈ initial mass | Conservation |

---

## 3. Dispersion Test Cases

### 3.1 Gaussian Plume

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_centerline_concentration_decreases_with_distance` | C(x₁) > C(x₂) for x₁ < x₂ | Decreasing | Physical (dilution) |
| `test_concentration_zero_at_source` | C(0,0,0) = 0 | Zero | Mathematical |
| `test_concentration_negative_x` | C(-x) = 0 | Zero | Physical (upwind) |
| `test_max_concentration_at_centerline` | C(y=0) > C(y≠0) | Maximum at centerline | Mathematical |
| `test_concentration_symmetric_about_centerline` | C(y) = C(-y) | Symmetric | Mathematical |
| `test_inverse_proportional_to_wind_speed` | C(u=2) > C(u=10) | C ∝ 1/u | Physical |
| `test_proportional_to_source_rate` | C(Q=2) = 2·C(Q=1) | Linear scaling | Mathematical |

### 3.2 Gaussian Puff

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_puff_peak_decreases_with_time` | C_max(t) decreases | Decreasing | Physical (dispersion) |
| `test_puff_moves_downwind` | Peak position moves with u·t | x_center = u·t | Physical |

### 3.3 Dense Gas

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_dense_gas_radius_increases` | Radius grows during slumping | R(t) increasing | Physical |
| `test_dense_gas_height_decreases` | Slumping spreads cloud | H(t) decreasing | Physical |
| `test_density_ratio_approaches_one` | Dilution by air entrainment | ρ_ratio → 1 | Physical |

---

## 4. Fire Test Cases

### 4.1 Pool Fire

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_burning_rate_positive_for_gasoline` | Gasoline burns | mdot > 0 | Sanity |
| `test_thermal_radiation_decreases_with_distance` | q(50) > q(100) > q(200) | Decreasing | Physical |
| `test_distance_to_4kw_less_than_1kw` | d(4 kW/m²) < d(1 kW/m²) | Ordered | Physical |
| `test_larger_pool_more_radiation` | Larger D → higher q at same R | q(D=20) > q(D=5) | Physical |

### 4.2 Jet Fire

| Test | Description | Expected | Type |
|------|-------------|----------|------|
| `test_flame_length_positive` | L > 0 | Positive | Sanity |
| `test_higher_mass_flow_longer_flame` | L(5 kg/s) > L(0.5 kg/s) | Increasing with mdot | Physical |

### 4.3 BLEVE

| Test | Description | Expected | Benchmark |
|------|-------------|----------|-----------|
| `test_fireball_diameter_roberts` | D = 5.8·M⁰·³³³ for 1000 kg | ≈58 m | Roberts (1981) |
| `test_fireball_duration_positive` | t_fb > 0 | Positive | CCPS |
| `test_fireball_diameter_increases_with_mass` | D ∝ M^(1/3) | Monotonic | Roberts |
| `test_bleve_radiation_decreases_with_distance` | q(100) > q(200) > q(500) | Decreasing | Physical |

---

## 5. Explosion Test Cases

### 5.1 TNT Equivalency

| Test | Description | Expected | Benchmark |
|------|-------------|----------|-----------|
| `test_w_tnt_calculation` | W = η·M·ΔHc/ΔHc_TNT | Computed value | Formula check |
| `test_overpressure_decreases_with_distance` | p(50) > p(100) > p(200) | Decreasing | Physical |
| `test_scaled_distance_relationship` | Same Z → same P (different M,R) | Within 20% | Kingery-Bulmash |
| `test_larger_mass_greater_overpressure` | M=5000 > M=100 at same R | Higher P | Physical |

### 5.2 TNO Multi-Energy

| Test | Description | Expected | Benchmark |
|------|-------------|----------|-----------|
| `test_strength_10_greater_than_4` | S=10 → higher P than S=4 | P(S10) > P(S4) | TNO curves |
| `test_auto_blast_strength_in_range` | Any congestion/confinement | 1 ≤ S ≤ 10 | Specification |

### 5.3 BST

| Test | Description | Expected | Benchmark |
|------|-------------|----------|-----------|
| `test_high_reactivity_greater_than_low` | High > Low reactivity | P(high) > P(low) | BST method |
| `test_mach_number_effect` | M=1.0 > M=0.1 | P(M1) > P(M0.1) | Physical |

---

## 6. Vulnerability Test Cases

### 6.1 Probit Core

| Test | Description | Expected | Benchmark |
|------|-------------|----------|-----------|
| `test_probit_5_is_50_percent` | P(5.0) | 0.5 | Gaussian CDF |
| `test_probit_zero_near_zero` | P(0) | < 0.001 | Gaussian CDF |
| `test_probit_ten_near_one` | P(10) | > 0.999 | Gaussian CDF |
| `test_probit_monotonically_increasing` | P(Y) non-decreasing | Monotonic | Mathematical |
| `test_probability_to_probit_roundtrip` | probit→P→probit | Within 0.01 | Mathematical |
| `test_probit_symmetry` | P(5+d) + P(5-d) = 1 | Symmetric | Gaussian |

### 6.2 Thermal Probit

| Test | Description | Expected |
|------|-------------|----------|
| `test_higher_heat_flux_higher_probability` | q=37.5 > q=5 kW/m² | P(37.5) > P(5) |
| `test_longer_exposure_higher_probability` | t=60s > t=10s | P(60) > P(10) |
| `test_zero_heat_flux_zero_probability` | q=0 | P ≈ 0 |

### 6.3 Shelter Factor

| Test | Description | Expected |
|------|-------------|----------|
| `test_indoor_less_than_outdoor` | C_in < C_out | Physical (infiltration) |
| `test_shelter_factor_positive_less_than_one` | 0 < SF < 1 | Physical |
| `test_outdoor_shelter_factor_is_one` | SF_outdoor = 1.0 | No protection |

---

## 7. QRA Test Cases

| Test | Description | Expected |
|------|-------------|----------|
| `test_create_generic_vessel_tree` | Tree created | Non-null EventTree |
| `test_small_leak_more_frequent_than_rupture` | Freq(small) > Freq(rupture) | Equipment data |
| `test_frequency_classification` | 1e-2 → FREQUENT, 1e-8 → EXTREMELY_REMOTE | ISO 17776 |
| `test_ignition_probability_in_range` | All in [0, 1] | Mathematical |
| `test_fn_decreasing` | F(N) non-increasing | Societal risk |
| `test_frequent_catastrophic_is_extreme` | HI+H5 → EXTREME | Risk matrix |

---

## 8. EoS Test Cases

| Test | Description | Expected |
|------|-------------|----------|
| `test_pr_z_factor_methane_300k_1mpa_positive` | Z > 0 | Physical |
| `test_pr_z_factor_vs_coolprop` | Within 10% of CoolProp | Reference (skipped if CoolProp unavailable) |
| `test_pr_density_positive` | ρ > 0 | Physical |
| `test_mixture_z_between_pure_components` | Z_mix between Z_pure | Physical |
| `test_srk_vs_pr_z_factor_similar` | Within 5% agreement | Cross-model |

---

## 9. Meteorology Test Cases

| Test | Description | Expected |
|------|-------------|----------|
| `test_strong_solar_day_light_wind_is_unstable` | Solar=800W/m², WS=2, Day → A/B | Pasquill |
| `test_overcast_strong_wind_is_neutral` | Overcast, WS=7 → D | Pasquill |
| `test_clear_night_light_wind_is_stable` | Clear night, WS=2 → E/F | Pasquill |
| `test_sigma_increases_with_distance` | σ(1000m) > σ(100m) | Physical |
| `test_u_50m_greater_than_u_10m_neutral` | u(50m) > u(10m) | Wind profile |
| `test_power_law_exponent_stable_greater_than_unstable` | p(F) > p(D) | Stability effect |
| `test_air_density_at_stp` | ρ ≈ 1.2 kg/m³ | STP |

---

## 10. Known Limitations

### 10.1 Model Limitations

- **Gaussian models** assume flat terrain, uniform wind — not suitable for
  complex terrain or urban canyons
- **Pool fire** burning rate correlation is empirical; may differ ±30% from
  measured values for specific fuels
- **TNT equivalency** efficiency factor (η) selection is judgment-based;
  for VCEs, use TNO or BST instead
- **Two-phase discharge** uses homogeneous equilibrium model (HEM);
  slip-ratio models may be more accurate for long pipes
- **Dense gas** model uses simplified slab approach; for detailed analysis,
  use CFD (e.g., FLACS, CHARM)
- **Probit functions** are generic population averages; individual
  susceptibility varies significantly
- **Ignition probabilities** are derived from limited incident data;
  site-specific factors not captured

### 10.2 Implementation Limitations

- **EoS validation** against CoolProp is skipped when CoolProp not installed;
  CoolProp is an optional dependency with complex build requirements on some platforms
- **Event tree** branch probabilities are user-input; no built-in reliability
  database for safety barriers
- **Individual risk** integration assumes spatial independence of scenarios
- **Wind profile** uses power law with constant exponent; roughness change
  effects not modeled
- **Pool evaporation** uses constant mass transfer coefficient; transient
  pool spread uses averaged correlations

---

## 11. Comparison with Published Benchmarks

| Model | Benchmark Reference | Comparison Method | Status |
|-------|-------------------|-------------------|--------|
| Orifice discharge | API 520 Annex B examples | Known mass flow cases | ✅ Order-of-magnitude |
| PSV sizing | API 520 Sizing examples | Required area comparison | ✅ Orifice designation |
| Pool fire | Mudan (1984) data | Burning rate order of magnitude | ✅ Directional |
| BLEVE diameter | Roberts (1981) correlation | Exact formula check | ✅ Exact match |
| TNT overpressure | Kingery-Bulmash (1984) | Scaled distance independence | ✅ Within 20% |
| Probit | Gaussian CDF | P(5)=0.5, monotonicity | ✅ Exact |
| EoS (PR) | CoolProp | Z-factor <10% deviation | ✅ (if CoolProp) |
| Stability | Pasquill (1961) | Known cases | ✅ Qualitative |
| Wind profile | Power law | u(z₂) > u(z₁) for z₂ > z₁ | ✅ Directional |

---

## 12. Running the Test Suite

```bash
# Run all validation tests
cd rekarisk
pytest tests/ -v

# Run specific test module
pytest tests/test_source_term.py -v

# Run with coverage report
pytest tests/ --cov=rekarisk --cov-report=term
```

## 13. Adding New Test Cases

Test cases should follow these guidelines:

1. **Physical checks:** Verify directionality (should increase/decrease), sign (should be positive), monotonicity
2. **Mathematical checks:** Verify formula correctness, symmetry, conservation
3. **Benchmark checks:** Compare against known published values
4. **Sanity checks:** Verify outputs are in sensible ranges

All test data should be inline (no external test data files).
Use `pytest.approx()` for floating-point comparisons.
Use `pytest.skip()` for optional dependencies (CoolProp).
