#!/usr/bin/env python3
"""QRA Pipeline with 7 ISO Sections — Realistic Indonesian Gas Plant Layout.

Based on a typical onshore natural gas processing facility:
- Process Area (compressor, separator, columns)
- Storage Tank Farm (condensate)
- Loading/Unloading Area
- Utility Area (power gen, instrument air)
- Pipeline Station (inlet/outlet pigging)
- Flare & KO Drum
- Control Room / Admin Building

Layout coordinates (meters):
  N (y+) ↑
  |
  |  [Flare/KO] (100, 120)
  |  [Utility] (-80, 60)     [Process] (0, 30)
  |  [Pipeline] (-40, 0)     [CtrlRoom] (20, -40)
  |                          [Storage] (80, -20)  [Loading] (110, -50)
  +----→ E (x+)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rekarisk.models.qra.qra_pipeline import (
    QRAPipeline, IsoSection, HoleSize, WeatherScenario,
    ReceptorPoint, WorkerGroup, DEFAULT_HOLE_SIZES,
)

# ── 7 ISO Sections — Realistic gas plant ─────────────────────────────────
iso_sections = [
    # 1. Process Area — compressor, separator, glycol contactor, filters, columns
    #    5 major equipment items, high-pressure natural gas
    IsoSection(name="Process Area", P=60e5, T=320.0,
               volume=8.5, composition="natural_gas", molecular_weight=20.5,
               fill_fraction=0.0, x=0, y=30, elevation=3.0, n_equipment=5,
               freq_scale=2.4),

    # 2. Storage Tank Farm — 2 condensate tanks
    IsoSection(name="Storage Tank Farm", P=3e5, T=305.0,
               volume=500.0, composition="propane", molecular_weight=44.1,
               fill_fraction=0.75, x=80, y=-20, elevation=0.5,
               rho_liquid=520.0, n_equipment=2),

    # 3. Loading/Unloading Area — truck loading bay
    IsoSection(name="Loading Area", P=5e5, T=300.0,
               volume=2.0, composition="propane", molecular_weight=44.1,
               fill_fraction=0.5, x=110, y=-50, elevation=0.0,
               rho_liquid=520.0, n_equipment=1),

    # 4. Utility Area — power generator, instrument air compressor, diesel tank
    IsoSection(name="Utility Area", P=15e5, T=300.0,
               volume=3.0, composition="natural_gas", molecular_weight=20.5,
               fill_fraction=0.0, x=-80, y=60, elevation=0.0, n_equipment=2),

    # 5. Pipeline Station — inlet pig receiver + outlet pig launcher
    IsoSection(name="Pipeline Station", P=70e5, T=315.0,
               volume=15.0, composition="natural_gas", molecular_weight=20.5,
               fill_fraction=0.0, x=-40, y=0, elevation=0.0, n_equipment=2),

    # 6. Flare & KO Drum — relief gas, low pressure
    IsoSection(name="Flare KO Drum", P=5e5, T=310.0,
               volume=12.0, composition="natural_gas", molecular_weight=20.5,
               fill_fraction=0.3, x=100, y=120, elevation=15.0),

    # 7. Control Room/Admin — office buildings, no process equipment
    IsoSection(name="Control Room", P=1e5, T=300.0,
               volume=1.0, composition="natural_gas", molecular_weight=20.5,
               fill_fraction=0.0, x=20, y=-40, elevation=0.0, n_equipment=0),
]

# ── Hole Sizes (standard OGP) ────────────────────────────────────────────
hole_sizes = [
    HoleSize("Small", 0.0064),
    HoleSize("Medium", 0.0254),
    HoleSize("Large", 0.1016),
    HoleSize("Fullbore", 0.2032),
]

# ── Weather Scenarios — East Kalimantan tropical ─────────────────────────
weather = [
    WeatherScenario(name="1.5F", wind_speed=1.5, stability_class="F",
                    probability=0.15, ambient_temperature=301.0, relative_humidity=0.85),
    WeatherScenario(name="3.5D", wind_speed=3.5, stability_class="D",
                    probability=0.55, ambient_temperature=302.0, relative_humidity=0.80),
    WeatherScenario(name="7.0C", wind_speed=7.0, stability_class="C",
                    probability=0.30, ambient_temperature=303.0, relative_humidity=0.75),
]

# ── Receptor Grid — 11 locations matching SAFETI NKT QRA ─────────────────
receptors = [
    ReceptorPoint(label="Process Area NKT", x=0, y=30),
    ReceptorPoint(label="Process Area CPPG North", x=0, y=55),
    ReceptorPoint(label="Process Area CPPG South", x=0, y=5),
    ReceptorPoint(label="Control Room NKT", x=20, y=-40),
    ReceptorPoint(label="Control Room CPPG", x=20, y=-30),
    ReceptorPoint(label="Storage Tank Farm", x=80, y=-20),
    ReceptorPoint(label="Loading Area", x=110, y=-50),
    ReceptorPoint(label="Substation Building", x=-10, y=-20),
    ReceptorPoint(label="Utility Area", x=-80, y=60),
    ReceptorPoint(label="Pipeline Station", x=-40, y=0),
    ReceptorPoint(label="Flare Area", x=100, y=120),
    ReceptorPoint(label="Security & Guard West", x=-90, y=-10),
    ReceptorPoint(label="Security & Guard North", x=-60, y=80),
]

# ── Worker Groups — 60 workers total ──────────────────────────────────────
workers = [
    WorkerGroup(name="Operator NKT", count=4,
                locations=[(0, 30, 0.40), (80, -20, 0.15), (20, -40, 0.15), (-80, 60, 0.10), (110, -50, 0.10)]),
    WorkerGroup(name="Sr Operator DCS", count=2,
                locations=[(20, -40, 0.70), (0, 30, 0.15), (80, -20, 0.10)]),
    WorkerGroup(name="Field Operator", count=3,
                locations=[(0, 30, 0.35), (80, -20, 0.20), (-40, 0, 0.15), (-80, 60, 0.15), (110, -50, 0.10)]),
    WorkerGroup(name="Maintenance Tech", count=3,
                locations=[(0, 30, 0.25), (-80, 60, 0.25), (80, -20, 0.20), (100, 120, 0.10), (110, -50, 0.10)]),
    WorkerGroup(name="Loading Operator", count=2,
                locations=[(110, -50, 0.50), (80, -20, 0.20), (0, 30, 0.15)]),
    WorkerGroup(name="Storage Operator", count=2,
                locations=[(80, -20, 0.50), (0, 30, 0.15), (110, -50, 0.15)]),
    WorkerGroup(name="Pipeline Technician", count=2,
                locations=[(-40, 0, 0.40), (0, 30, 0.20), (-80, 60, 0.15)]),
    WorkerGroup(name="Security Guard", count=4,
                locations=[(-90, -10, 0.35), (-60, 80, 0.35), (0, 30, 0.10)]),
    WorkerGroup(name="Admin Staff", count=4,
                locations=[(20, -40, 0.60), (20, -30, 0.20)]),
    WorkerGroup(name="Utility Operator", count=2,
                locations=[(-80, 60, 0.50), (0, 30, 0.15), (20, -40, 0.15)]),
    WorkerGroup(name="Flare Operator", count=1,
                locations=[(100, 120, 0.30), (0, 30, 0.30), (-80, 60, 0.15)]),
]

# ── Custom shelter factors for this plant ─────────────────────────────────
shelter = {
    "Process Area NKT": 1.0,
    "Process Area CPPG North": 1.0,
    "Process Area CPPG South": 1.0,
    "Control Room NKT": 0.2,
    "Control Room CPPG": 0.2,
    "Storage Tank Farm": 1.0,
    "Loading Area": 1.0,
    "Substation Building": 0.3,
    "Utility Area": 0.8,
    "Pipeline Station": 1.0,
    "Flare Area": 0.8,
    "Security & Guard West": 0.7,
    "Security & Guard North": 0.7,
}


def run():
    pipeline = QRAPipeline(
        iso_sections=iso_sections,
        hole_sizes=hole_sizes,
        weather_scenarios=weather,
        receptor_grid=receptors,
        worker_groups=workers,
        receptor_shelter_factors=shelter,
    )

    print("Running QRA with 7 ISO sections...")
    result = pipeline.run()

    # ── LSIR Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LSIR — 7 ISO Sections, 11 Receptor Points")
    print("=" * 70)

    safeti_targets = {
        "Process Area NKT": 1.86e-4,
        "Process Area CPPG North": 8.5e-5,
        "Process Area CPPG South": 1.2e-4,
        "Control Room NKT": 3.7e-5,
        "Storage Tank Farm": 5.0e-5,
        "Loading Area": 3.0e-5,
        "Utility Area": 2.0e-5,
        "Pipeline Station": 4.0e-5,
        "Flare Area": 1.0e-5,
    }

    for (rx, ry), lsir_val in sorted(result.lsir_grid.items(),
                                      key=lambda x: -x[1]):
        rp = next((r for r in receptors if r.x == rx and r.y == ry), None)
        label = rp.label if rp else f"({rx},{ry})"
        line = f"  {label:35s}  LSIR = {lsir_val:.3e}/yr"
        if label in safeti_targets:
            target = safeti_targets[label]
            ratio = lsir_val / target if target > 0 else float('inf')
            line += f"  (SAFETI: {target:.2e}, ratio: {ratio:.2f}x)"
        print(line)

    # ── IRPA Results ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("IRPA — Individual Risk per Annum")
    print("-" * 70)

    total_workers = sum(w.count for w in workers)
    for name, irpa in sorted(result.irpa_table.items(), key=lambda x: -x[1]):
        alarp = result.alarp.get(name, "?")
        print(f"  {name:30s}  IRPA = {irpa:.3e}/yr  [{alarp}]")

    # ── PLL & Summary ─────────────────────────────────────────────────────
    print(f"\n  Total Workers: {total_workers}")
    print(f"  PLL (total):   {result.pll_total:.3e}/yr")
    print(f"  Scenarios:     {result.scenario_count}")

    # ── Dominant Scenarios ────────────────────────────────────────────────
    print("\n  Top 5 Risk Contributors:")
    for d in result.dominant:
        print(f"    {d['scenario']:50s}  f={d['frequency']:.2e}/yr")

    if result.warnings:
        print(f"\n  Warnings: {len(result.warnings)}")
        for w in result.warnings[:5]:
            print(f"    {w}")

    print("\n✅ QRA Pipeline with 7 ISO sections complete!")
    return result


if __name__ == "__main__":
    run()
