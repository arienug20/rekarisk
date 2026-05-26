"""
Validation tests for Rekarisk Phase 9 — Vulnerability Module.
"""
import sys
sys.path.insert(0, "src")

from rekarisk.models.vulnerability.probit import (
    probit_to_probability, probability_to_probit,
    thermal_probit, overpressure_probit, toxic_probit,
    ThermalModel, OverpressureModel,
)
from rekarisk.models.vulnerability.toxic_dose import (
    toxic_load, toxic_load_from_time_series,
    compare_to_erpg, compare_to_aegl,
)
from rekarisk.models.vulnerability.shelter_factor import (
    indoor_concentration, shelter_factor, time_to_reach,
)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    emoji = "OK" if condition else "!!"
    print(f"  [{emoji}] {name}: {detail}")
    if condition:
        passed += 1
    else:
        failed += 1

print("=" * 60)
print("REKARISK PHASE 9 — VULNERABILITY MODULE VALIDATION")
print("=" * 60)

# --- Probit Core ---
print("\n--- Probit Core ---")
p = probit_to_probability(5.0)
check("probit_to_probability(5.0) = 0.50", abs(p - 0.5) < 1e-6, f"got {p:.6f}")

p0 = probit_to_probability(0)
check("probit_to_probability(0) ~ 0", p0 < 0.001, f"got {p0:.6e}")

p10 = probit_to_probability(10)
check("probit_to_probability(10) ~ 1", p10 > 0.999, f"got {p10:.6f}")

y = probability_to_probit(0.5)
check("probability_to_probit(0.5) = 5.0", abs(y - 5.0) < 1e-6, f"got {y:.6f}")

# --- Thermal ---
print("\n--- Thermal Probit ---")
Y_t, P_t = thermal_probit(20000, 60, ThermalModel.EISENBERG)
check("Thermal 20kW/m2 60s (Eisenberg): P > 0.5", P_t > 0.5, f"Y={Y_t:.3f}, P={P_t:.4f}")

Y_tno, P_tno = thermal_probit(20000, 60, ThermalModel.TNO)
check("Thermal 20kW/m2 60s (TNO): P > 0", P_tno > 0, f"Y={Y_tno:.3f}, P={P_tno:.4f}")

Y_lees, P_lees = thermal_probit(20000, 60, ThermalModel.LEES)
check("Thermal 20kW/m2 60s (Lees): P > 0", P_lees > 0, f"Y={Y_lees:.3f}, P={P_lees:.4f}")

# --- Overpressure ---
print("\n--- Overpressure Probit ---")
Y_op, P_op = overpressure_probit(100000, OverpressureModel.EISENBERG_LUNG)
check("Overpressure 100kPa (Eisenberg lung): P > 0.9", P_op > 0.9, f"Y={Y_op:.3f}, P={P_op:.4f}")

Y_op2, P_op2 = overpressure_probit(10000, OverpressureModel.TNO_STRUCTURAL)
check("Overpressure 10kPa (TNO structural): P > 0", P_op2 > 0, f"Y={Y_op2:.3f}, P={P_op2:.4f}")

Y_op3, P_op3 = overpressure_probit(5000, OverpressureModel.TNO_STRUCTURAL)
check("Overpressure 5kPa (TNO structural): P < 1", P_op3 < 1.0, f"Y={Y_op3:.3f}, P={P_op3:.6f}")

# --- Toxic ---
print("\n--- Toxic Probit ---")
for substance, expected_min in [("chlorine", 0.01), ("ammonia", 0.001), 
                                   ("hydrogen_sulfide", 0.01), ("hydrogen_chloride", 0.1),
                                   ("phosgene", 0.5)]:
    Y_s, P_s = toxic_probit(100, 30, substance)
    check(f"Toxic {substance} 100ppm 30min: P > {expected_min}",
          P_s > expected_min, f"Y={Y_s:.3f}, P={P_s:.4f}")

# Low concentration - should be low probability
Y_low, P_low = toxic_probit(1, 10, "chlorine")
check("Toxic Cl2 1ppm 10min: P < 0.5", P_low < 0.5, f"Y={Y_low:.3f}, P={P_low:.6f}")

# --- Toxic Dose ---
print("\n--- Toxic Dose ---")
tl = toxic_load(100, 30, 1.0)
check("toxic_load(100,30,n=1) = 3000", abs(tl - 3000) < 1, f"got {tl:.1f}")

tl2 = toxic_load(100, 30, 2.75)
check("toxic_load(100,30,n=2.75) > 0", tl2 > 0, f"got {tl2:.1f}")

# Time series integration
ts = [(0, 0), (100, 10), (100, 20), (50, 30)]
tl_ts = toxic_load_from_time_series(ts, 1.0)
check("toxic_load_from_time_series", tl_ts > 0, f"got {tl_ts:.1f}")

# ERPG comparison
erpg = compare_to_erpg(50, "chlorine")
check("ERPG: 50ppm Cl2 exceeds ERPG-3", erpg["exceeds_erpg3"], f"{erpg}")

# AEGL comparison
aegl = compare_to_aegl(10, 60, "chlorine")
check("AEGL: 10ppm Cl2 60min exceeds AEGL-2", aegl["exceeds_aegl2"], f"{aegl}")

# --- Shelter Factor ---
print("\n--- Shelter Factor ---")
C_in = indoor_concentration(100, 10, ach=1.0)
check("Shelter: C_out=100, t=10min, ACH=1 → C_in < 100", C_in < 100, f"C_in={C_in:.2f}")

C_in2 = indoor_concentration(100, 60, ach=0.5)
check("Shelter: ACH=0.5, t=60min → C_in < 100", C_in2 < 100, f"C_in={C_in2:.2f}")

sf = shelter_factor(100, 10, ach=1.0)
check("Shelter factor: SF < 1 at short time", sf < 1.0, f"SF={sf:.4f}")

C_no_vent = indoor_concentration(100, 60, ach=0.01)
check("Shelter: ACH=0.01, t=60min → very low infiltration", C_no_vent < 15, f"C_in={C_no_vent:.2f}")

t_reach = time_to_reach(100, 50, ach=1.0)
check("time_to_reach: C_out=100, target=50, ACH=1", t_reach is not None and t_reach > 0, f"t={t_reach:.1f} min" if t_reach else "None")

# --- Summary ---
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed", end="")
if failed > 0:
    print(f", {failed} FAILED")
    sys.exit(1)
else:
    print(" — ALL TESTS PASSED!")
