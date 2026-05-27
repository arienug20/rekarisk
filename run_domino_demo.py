"""
Rekarisk — Domino Effect Analysis Demo.

Scenario: Propane storage area at a gas plant with multiple vessels.
Primary event: Pool fire at TK-301 (propane storage tank).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
from rekarisk.models.qra.domino import (
    EquipmentType, EscalationVector, DamageLevel, SubstanceCategory,
    Equipment, PrimaryEvent,
    run_domino_analysis, plot_domino_map, plot_escalation_summary, plot_domino_chain,
)

OUTDIR = Path("/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("REKARISK — DOMINO EFFECT ANALYSIS")
print("Propane Storage Area — Gas Plant")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# Equipment Layout
# ══════════════════════════════════════════════════════════════════════════════
equipment = [
    Equipment(
        id="TK-301", name="Propane Storage Tank",
        equipment_type=EquipmentType.ATMOSPHERIC_TANK,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=50000, x=0, y=0, diameter=10, height=8,
        operating_pressure=8, design_pressure=10,
        has_bund=True, bund_radius=12,
    ),
    Equipment(
        id="V-302", name="Propane Separator",
        equipment_type=EquipmentType.PRESSURE_VESSEL,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=5000, x=30, y=10, diameter=2, height=6,
        operating_pressure=12, design_pressure=15,
        is_insulated=True,
    ),
    Equipment(
        id="V-303", name="Butane Storage Vessel",
        equipment_type=EquipmentType.PRESSURE_VESSEL,
        substance="Butane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=20000, x=40, y=-15, diameter=4, height=7,
        operating_pressure=5, design_pressure=8,
    ),
    Equipment(
        id="C-304", name="De-propanizer Column",
        equipment_type=EquipmentType.COLUMN,
        substance="LPG Mix", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=8000, x=-20, y=25, diameter=2, height=20,
        operating_pressure=15, design_pressure=20,
        has_deluge=True,
    ),
    Equipment(
        id="TK-305", name="Condensate Storage",
        equipment_type=EquipmentType.ATMOSPHERIC_TANK,
        substance="Condensate", substance_category=SubstanceCategory.FLAMMABLE_LIQUID,
        inventory_kg=100000, x=-30, y=-20, diameter=12, height=10,
        operating_pressure=0.1, design_pressure=0.5,
        has_bund=True, bund_radius=15,
    ),
    Equipment(
        id="HX-306", name="Propane Chiller",
        equipment_type=EquipmentType.HEAT_EXCHANGER,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=2000, x=15, y=25, diameter=1, height=3,
        operating_pressure=10, design_pressure=14,
    ),
    Equipment(
        id="P-307A", name="Propane Transfer Pump A",
        equipment_type=EquipmentType.PUMP,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=500, x=15, y=-5, diameter=0.5, height=1.5,
        operating_pressure=10, design_pressure=15,
    ),
    Equipment(
        id="P-307B", name="Propane Transfer Pump B",
        equipment_type=EquipmentType.PUMP,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=500, x=20, y=5, diameter=0.5, height=1.5,
        operating_pressure=10, design_pressure=15,
    ),
    Equipment(
        id="K-308", name="Propane Compressor",
        equipment_type=EquipmentType.COMPRESSOR,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_GAS,
        inventory_kg=1000, x=-10, y=15, diameter=2, height=3,
        operating_pressure=15, design_pressure=20,
    ),
    Equipment(
        id="FIN-309", name="Air Cooler",
        equipment_type=EquipmentType.FIN_FAN_COOLER,
        substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
        inventory_kg=1500, x=5, y=-15, diameter=3, height=2,
        operating_pressure=12, design_pressure=16,
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# Primary Event: Pool Fire at TK-301
# ══════════════════════════════════════════════════════════════════════════════
primary = PrimaryEvent(
    equipment_id="TK-301",
    event_type="pool_fire",
    frequency=5e-6,
    thermal_power_kw=150000,  # 150 MW pool fire
    tnt_mass_kg=250,          # VCE from 10% cloud
    fireball_radius_m=50,
    source_height_m=4.0,
    pool_radius_m=12,
)

print(f"\n📋 Primary Event: {primary.event_type.upper()} at {primary.equipment_id}")
print(f"   Frequency: {primary.frequency:.1e}/yr")
print(f"   Thermal power: {primary.thermal_power_kw/1000:.0f} MW")
print(f"   TNT equivalent: {primary.tnt_mass_kg:.0f} kg")
print(f"\n📍 Equipment Layout: {len(equipment)} items")

# ══════════════════════════════════════════════════════════════════════════════
# Run Domino Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔄 Running domino analysis (max 3 escalation orders)...\n")

result = run_domino_analysis(
    primary_event=primary,
    equipment_list=equipment,
    max_escalation_order=3,
    response_time_min=10.0,
)

# Print results
print("═" * 60)
print("ESCALATION ANALYSIS RESULTS")
print("═" * 60)

print(f"\n📊 Summary:")
print(f"   Total equipment: {result.summary['total_equipment']}")
print(f"   Escalation links found: {result.summary['total_escalation_links']}")
print(f"   Significant links: {result.summary['significant_links']}")
print(f"   Domino scenarios: {result.summary['domino_scenarios']}")
print(f"   Max cascade order: {result.summary['max_cascade_order']}")
print(f"   Max escalation distance: {result.summary['max_escalation_distance_m']:.0f} m")
print(f"   Equipment at risk: {', '.join(result.summary['equipment_at_risk'])}")

print(f"\n🔗 Escalation Links:")
print(f"   {'Target':<12} | {'Dist (m)':>8} | {'Vector':>18} | {'Intensity':>12} | {'Damage':>12} | {'P(esc)':>8} | {'TTF (min)':>9}")
print("   " + "-" * 95)

significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
for link in sorted(significant, key=lambda l: -l.escalation_prob):
    unit = "kW/m²" if link.vector == EscalationVector.THERMAL_RADIATION else "kPa"
    print(f"   {link.target_id:<12} | {link.distance_m:>8.1f} | {link.vector.value:>18} | {link.intensity:>8.1f} {unit} | {link.damage_level.value:>12} | {link.escalation_prob:>7.1%} | {link.ttf_minutes:>8.1f}")

print(f"\n💥 Domino Scenarios:")
for i, scenario in enumerate(sorted(result.domino_scenarios, key=lambda s: -s.total_frequency), 1):
    chain = " → ".join(scenario.chain)
    print(f"   #{i}: {chain}")
    print(f"       f = {scenario.total_frequency:.2e}/yr | Inventory = {scenario.total_inventory_released:.0f} kg | Order = {scenario.max_order}")

# ══════════════════════════════════════════════════════════════════════════════
# Generate Plots
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n📊 Generating plots...")

plot_domino_map(result, save_path=str(OUTDIR / "09_domino_escalation_map.png"))
plot_escalation_summary(result, save_path=str(OUTDIR / "10_domino_summary.png"))
plot_domino_chain(result, save_path=str(OUTDIR / "11_domino_chain.png"))

print("\n✅ Domino analysis complete!")
