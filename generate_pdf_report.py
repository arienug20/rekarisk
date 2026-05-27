"""
Generate PDF Report from Propane Risk Scenario.
Uses rekarisk.report.pdf_generator.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
from datetime import datetime

from rekarisk.report.pdf_generator import generate_report

OUTDIR = Path("/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario")
PDF_PATH = OUTDIR / "Propane_Vessel_Leak_Risk_Assessment.pdf"

project_data = {
    "name": "Propane Storage Vessel Leak — Risk Assessment",
    "author": "Arie Nugraha",
    "created_at": datetime.now().strftime("%d %B %Y"),
    "organization": "PT Rekayasa Engineering",
    "format_version": "1.0",
    "weather_cases": [
        {
            "name": "Normal (Stability D)",
            "wind_speed": 2.5,
            "wind_direction": 225,
            "temperature": 30,
            "ambient_temp": 30,
            "stability_class": "D",
            "humidity": 75,
            "roughness_length": 1.0,
        },
        {
            "name": "Worst Case (Stability F)",
            "wind_speed": 1.5,
            "wind_direction": 225,
            "temperature": 25,
            "ambient_temp": 25,
            "stability_class": "F",
            "humidity": 85,
            "roughness_length": 1.0,
        },
    ],
    "substances": [
        {"name": "Propane (C₃H₈)", "cas": "74-98-6"},
    ],
}

executive_summary = (
    "This report presents a quantitative risk assessment (QRA) for a propane storage "
    "vessel leak scenario at a gas processing facility in Balikpapan, East Kalimantan. "
    "The analysis evaluates the consequences of a 25 mm hole in a 10 m³ propane storage "
    "sphere operating at 8 bar gauge pressure.<br/><br/>"
    "The analysis covers the full consequence chain: source term characterization, "
    "gas dispersion modeling, fire consequences (pool fire, jet fire, BLEVE), "
    "explosion modeling (TNT Equivalency, TNO Multi-Energy, Baker-Strehlow-Tang), "
    "vulnerability assessment using probit functions, and quantitative risk assessment "
    "including individual risk contours, societal risk (FN curves), and risk matrix "
    "classification.<br/><br/>"
    "<b>Key Findings:</b><br/>"
    "• The propane release rate through a 25 mm hole at 8 bar is <b>0.65 kg/s</b> (choked flow).<br/>"
    "• Vessel blowdown releases <b>122.6 kg</b> of propane in approximately <b>1.9 minutes</b>.<br/>"
    "• Pool fire (8 m diameter) produces thermal radiation of 66 kW/m² (SEP) with flame height of 6.8 m.<br/>"
    "• BLEVE fireball radius is <b>50 m</b> with duration of <b>5.7 seconds</b>.<br/>"
    "• At 50 m from pool fire, fatality probability from thermal exposure is <b>100%</b>.<br/>"
    "• At 100 m, thermal fatality probability drops to <b>97.7%</b> and is negligible beyond 200 m.<br/>"
    "• The societal risk FN curve falls within the <b>ALARP region</b> of the Dutch BEVI criteria.<br/>"
    "• Recommended mitigation: install gas detection within 50 m, emergency shutdown system, "
    "and water spray/deluge system for the propane storage area."
)

conclusion = (
    "The consequence analysis demonstrates that a propane storage vessel leak poses "
    "significant thermal and explosion hazards within the immediate vicinity (&lt;100 m). "
    "The pool fire scenario presents the highest risk to nearby personnel, with near-certain "
    "fatality within 100 m and significant burn risk up to approximately 200 m.<br/><br/>"
    "<b>Recommendations:</b><br/>"
    "1. <b>Emergency Shutdown (ESD) System:</b> Install automated ESD valves on the propane "
    "vessel with activation triggered by gas detectors (target: isolate within 30 seconds).<br/>"
    "2. <b>Fire &amp; Gas Detection:</b> Deploy UV/IR flame detectors and catalytic bead gas "
    "detectors within 30 m of the propane storage, with audible/visual alarms.<br/>"
    "3. <b>Water Deluge System:</b> Install water spray/deluge system rated at 10.2 L/min/m² "
    "over the propane vessel and bund area per NFPA 15.<br/>"
    "4. <b>Bunding:</b> Ensure bund capacity ≥ 110% of largest vessel volume with slope to "
    "remote impounding basin (minimum 15 m from vessel).<br/>"
    "5. <b>Separation Distances:</b> Maintain minimum 100 m exclusion zone for occupied "
    "buildings from propane storage per API RP 752.<br/>"
    "6. <b>Emergency Response:</b> Develop pre-incident plan including evacuation routes "
    "for personnel within 500 m radius.<br/><br/>"
    "This assessment should be reviewed and updated when there are changes to the facility "
    "layout, inventory, operating conditions, or surrounding population density."
)

# Build results list
results = [
    # Source Term
    {
        "name": "Orifice Gas Release (25 mm hole, 8 bar)",
        "type": "source_term",
        "inputs": {
            "Hole diameter": "25 mm",
            "Upstream pressure": "8 bar gauge",
            "Downstream pressure": "1.013 bar (atmospheric)",
            "Discharge coefficient (Cd)": "0.62",
            "Fluid phase": "Gas (propane)",
            "Temperature": "25°C (298.15 K)",
        },
        "summary": {
            "Mass flow rate": "0.651 kg/s",
            "Exit velocity": "244.5 m/s",
            "Mass flux (G)": "1,326.0 kg/(m²·s)",
            "Flow regime": "Choked gas flow",
        },
        "plots": [
            {"path": str(OUTDIR / "01_blowdown_summary.png"),
             "caption": "Figure 1: Vessel Blowdown Profile — Pressure, Temperature, Mass, and Flow Rate vs Time"},
        ],
    },
    # Dispersion
    {
        "name": "Gaussian Plume Dispersion (Stability D & F)",
        "type": "dispersion",
        "inputs": {
            "Source rate": "0.651 kg/s (from orifice)",
            "Wind speed": "2.5 m/s (D) / 1.5 m/s (F)",
            "Stability class": "D (normal) and F (worst case)",
            "Release height": "1.0 m",
            "Terrain": "Urban",
            "Temperature": "30°C (D) / 25°C (F)",
        },
        "summary": {
            "Max centerline concentration (D)": "See plot",
            "Max centerline concentration (F)": "See plot (higher than D)",
            "Model": "Gaussian Plume (Pasquill-Gifford)",
        },
        "plots": [
            {"path": str(OUTDIR / "02_dispersion_profile.png"),
             "caption": "Figure 2: Propane Gas Dispersion — Centerline Concentration vs Distance (Stability D and F)"},
        ],
        "thresholds": {
            "LEL (1.8 vol% ≈ 0.033 kg/m³)": "~50 m (Stability D)",
            "0.5× LEL": "~80 m (Stability D)",
            "ERPG-2 (5500 mg/m³)": "~100 m (Stability D)",
            "ERPG-3 (17000 mg/m³)": "~40 m (Stability D)",
        },
    },
    # Fire
    {
        "name": "Pool Fire (8 m diameter propane pool)",
        "type": "fire",
        "inputs": {
            "Pool diameter": "8.0 m",
            "Substance": "Propane",
            "Radiative fraction": "0.35",
            "Wind speed": "2.5 m/s",
            "Ambient temperature": "30°C",
            "Relative humidity": "70%",
        },
        "summary": {
            "Flame length": "6.8 m",
            "Surface Emissive Power (SEP)": "66.0 kW/m²",
            "Total burning rate": "2.76 kg/s",
            "Fire type": "Pool Fire",
        },
        "plots": [
            {"path": str(OUTDIR / "03_pool_fire_radiation.png"),
             "caption": "Figure 3: Pool Fire Thermal Radiation vs Distance with Damage Thresholds"},
        ],
        "thresholds": {
            "37.5 kW/m² (Equipment damage)": "~30 m",
            "12.5 kW/m² (Minor burn 30s)": "~55 m",
            "4.0 kW/m² (Pain threshold)": "~100 m",
            "1.6 kW/m² (Safe/No discomfort)": "~160 m",
        },
    },
    # Explosion
    {
        "name": "Vapor Cloud Explosion (Delayed Ignition)",
        "type": "explosion",
        "inputs": {
            "Mass in flammable cloud": "12.3 kg (10% of released inventory)",
            "Heat of combustion": "50.35 MJ/kg",
            "TNT explosion efficiency": "5%",
            "TNO blast strength": "7",
            "BST fuel reactivity": "High",
            "BST confinement": "2D (open terrain)",
            "BST congestion": "Medium",
        },
        "summary": {
            "TNT equivalency": "Calculated (5% η)",
            "TNO Multi-Energy": "Blast strength 7",
            "Baker-Strehlow-Tang": "High reactivity, 2D, medium congestion",
        },
        "plots": [
            {"path": str(OUTDIR / "04_explosion_overpressure.png"),
             "caption": "Figure 4: VCE Overpressure vs Distance — Three Methods Compared"},
        ],
        "thresholds": {
            "140 kPa (Building collapse)": "~50 m",
            "21 kPa (Steel structure damage)": "~120 m",
            "7 kPa (Window breakage)": "~250 m",
            "2 kPa (Minor structural damage)": "~450 m",
        },
    },
    # Vulnerability
    {
        "name": "Vulnerability Assessment (Probit Analysis)",
        "type": "vulnerability",
        "inputs": {
            "Hazard types": "Thermal radiation (pool fire), Overpressure (explosion)",
            "Exposure time (thermal)": "60 seconds",
            "Probit model": "Eisenberg (thermal), PIET (overpressure)",
        },
        "summary": {
            "Thermal — 50 m": "P(fatality) = 100%",
            "Thermal — 100 m": "P(fatality) = 97.7%",
            "Thermal — 200 m": "P(fatality) = 0.3%",
            "Thermal — 500 m": "P(fatality) ≈ 0%",
            "Overpressure — 50 m": "P(fatality) ≈ 0% (low mass in cloud)",
        },
        "plots": [
            {"path": str(OUTDIR / "05_vulnerability_curves.png"),
             "caption": "Figure 5: Fatality Probability vs Distance — Thermal and Overpressure Vulnerability"},
        ],
        "table_headers": ["Distance (m)", "Q (kW/m²)", "Thermal P(fat)", "ΔP (kPa)", "Overp. P(fat)"],
        "table_rows": [
            ["50", "0.11", "100.0%", "3.77", "≈ 0%"],
            ["100", "0.03", "97.7%", "1.77", "≈ 0%"],
            ["200", "0.01", "0.3%", "0.85", "≈ 0%"],
            ["300", "0.00", "0.0%", "0.56", "≈ 0%"],
            ["500", "0.00", "0.0%", "0.34", "≈ 0%"],
        ],
    },
    # QRA
    {
        "name": "Quantitative Risk Assessment",
        "type": "qra",
        "inputs": {
            "Initiating event frequency": "5 × 10⁻⁶ /year",
            "Event tree outcomes": "Safe dispersal, Flash fire, VCE, Jet fire",
            "Risk criteria": "Dutch BEVI (ALARP)",
        },
        "summary": {
            "Safe dispersal": "3.15 × 10⁻⁶ /yr",
            "Flash fire": "8.10 × 10⁻⁷ /yr",
            "VCE explosion": "5.40 × 10⁻⁷ /yr",
            "Jet fire": "5.00 × 10⁻⁷ /yr",
        },
        "ir_thresholds": {
            "1×10⁻⁶ /yr (HSE UK tolerable)": "~180 m",
            "1×10⁻⁵ /yr (TNO acceptable)": "~100 m",
            "1×10⁻⁴ /yr (Intolerable public)": "~60 m",
        },
        "fn_data": {
            "n": [1, 3, 10, 30, 100, 200],
            "f": [5e-6, 8.1e-6, 9.1e-6, 9.6e-6, 9.7e-6, 9.75e-6],
        },
        "plots": [
            {"path": str(OUTDIR / "06_fn_curve.png"),
             "caption": "Figure 6: Societal Risk FN Curve with Dutch BEVI ALARP Criteria"},
            {"path": str(OUTDIR / "07_risk_matrix.png"),
             "caption": "Figure 7: 5×5 Risk Matrix (ISO 17776) with Scenario Classifications"},
            {"path": str(OUTDIR / "08_risk_contour_map.png"),
             "caption": "Figure 8: Individual Risk Contour Map Around Propane Storage Facility"},
        ],
    },
]

# Generate
print("Generating PDF report...")
output = generate_report(
    project_data=project_data,
    results=results,
    output_path=PDF_PATH,
    cover_info={
        "project_name": "Propane Storage Vessel Leak — Risk Assessment",
        "author": "Arie Nugraha",
        "date": datetime.now().strftime("%d %B %Y"),
        "version": "1.0",
        "organization": "PT Rekayasa Engineering",
    },
    executive_summary=executive_summary,
    conclusion=conclusion,
)

print(f"✅ PDF generated: {output}")
print(f"   Size: {os.path.getsize(output) / 1024:.0f} KB")
