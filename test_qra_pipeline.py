#!/usr/bin/env python3
"""Test the end-to-end QRA pipeline using NKT data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pipeline():
    from rekarisk.models.qra.qra_pipeline import (
        QRAPipeline, IsoSection, HoleSize, WeatherScenario, 
        ReceptorPoint, WorkerGroup, DEFAULT_HOLE_SIZES
    )

    iso_sections = [
        IsoSection(name="ISO 1", P=450.05*6894.76+101325, T=(90.1-32)*5/9+273.15,
                   volume=0.564, composition="natural_gas", molecular_weight=20.5,
                   fill_fraction=0.0),
        IsoSection(name="ISO 2G", P=450.05*6894.76+101325, T=(90.1-32)*5/9+273.15,
                   volume=2.6232, composition="natural_gas", molecular_weight=20.5,
                   fill_fraction=0.0),
        IsoSection(name="ISO 5", P=430.05*6894.76+101325, T=(88.7-32)*5/9+273.15,
                   volume=15.767, composition="natural_gas", molecular_weight=20.5,
                   fill_fraction=0.0),
    ]

    locations = [
        ReceptorPoint(label="Process Area NKT", x=0, y=0),
        ReceptorPoint(label="Control Room NKT", x=15, y=25),
        ReceptorPoint(label="Metering Area", x=25, y=-10),
        ReceptorPoint(label="Utility Area", x=-15, y=-50),
        ReceptorPoint(label="Support Area", x=-20, y=-30),
    ]

    workers = [
        WorkerGroup(name="Operator NKT", count=2,
                    locations=[(0, 0, 0.40), (25, -10, 0.15), (15, 25, 0.20), (-15, -50, 0.10), (-20, -30, 0.15)]),
        WorkerGroup(name="Sr Operator DCS", count=1,
                    locations=[(15, 25, 0.70), (0, 0, 0.20), (25, -10, 0.10)]),
        WorkerGroup(name="Field Operator", count=1,
                    locations=[(0, 0, 0.50), (25, -10, 0.20), (-15, -50, 0.30)]),
    ]

    weather = [
        WeatherScenario(name="1.35C", wind_speed=1.35, stability_class="C", probability=0.15, ambient_temperature=300.65, relative_humidity=0.8232),
        WeatherScenario(name="5.5D", wind_speed=5.5, stability_class="D", probability=0.85, ambient_temperature=300.65, relative_humidity=0.8232),
    ]

    holes = {"Small": 5, "Medium": 50, "Large": 100, "Fullbore": 152.4}

    pipeline = QRAPipeline(
        iso_sections=iso_sections,
        hole_sizes=[HoleSize(name=n, diameter=d/1000) for n, d in holes.items()],
        weather_scenarios=weather,
        receptor_grid=locations,
        worker_groups=workers,
    )

    result = pipeline.run()

    print("=" * 70)
    print("QRA PIPELINE — NKT Data")
    print("=" * 70)

    print("\nLSIR per Location:")
    # result.lsir is a dict of label -> lsir
    if hasattr(result, 'lsir_grid') and result.lsir_grid is not None:
        for pt, val in result.lsir_grid.items():
            print(f"  {pt}: {val:.2e}/year")
    elif hasattr(result, 'lsir'):
        for k, v in result.lsir.items():
            print(f"  {k}: {v:.2e}/year")

    print("\nIRPA per Worker:")
    if hasattr(result, 'irpa_table'):
        for w in result.irpa_table:
            print(f"  {w}")
    elif hasattr(result, 'irpa'):
        for w, irpa in result.irpa.items():
            print(f"  {w}: {irpa:.2e}/year")

    print(f"\nTotal PLL: {result.pll_total:.2e}/year")
    print(f"Scenarios: {result.scenario_count}")
    print("\nPipeline SUCCESS — all modules connected!")

if __name__ == "__main__":
    test_pipeline()
