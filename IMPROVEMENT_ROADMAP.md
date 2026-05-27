# Rekarisk Full Feature Audit & Improvement Roadmap
# Based on FERA (FNKT-20-P1-SR-006) and QRA (FNKT-20-P1-SR-007) comparison

## Codebase Stats: 82 Python files, 56,516 lines

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: CORE — Thermodynamics & Substance Database
# ══════════════════════════════════════════════════════════════════════════════

## 1.1 substance_db.py (395L, 60 substances)
PROBLEMS:
- Only 60 substances — FERA doc uses natural gas mixture (C1-C7+)
- No mixture composition support (only pure substances)
- Missing key H2S properties needed for toxic dispersion
- No molecular weight lookup from composition
IMPROVE:
- Add 50+ more substances (common oil & gas chemicals)
- Add mixture support: user defines mol% composition, auto-calc MW, LFL, UFL, LEL
- Add H2S, CO2, N2 with full DIPPR parameters
- Add typical natural gas/condensate compositions from field data

## 1.2 eos.py (994L, Peng-Robinson + SRK + Van der Waals)
PROBLEMS:
- Works for pure substances but mixture VLE not fully connected to substance_db
- No flash calculation (two-phase split) — needed to determine release phase
IMPROVE:
- Implement isothermal flash (Rachford-Rice) for mixtures
- Connect to phase_envelope.py for automatic phase detection
- This is CRITICAL: PHAST determines gas/liquid/two-phase from VLE

## 1.3 phase_envelope.py (901L)
PROBLEMS:
- Has dew/bubble point calculation but not connected to release scenarios
- QRA needs to know: "at T=32°C, P=31 barg, is this stream gas or liquid?"
IMPROVE:
- Add auto-phase-detection function: given (T, P, composition) → phase state
- Connect to orifice.py for automatic phase dispatch

## 1.4 dippr.py (516L, 12 equation types)
STATUS: Good foundation. Needs more substance data.
IMPROVE:
- Add DIPPR parameters for H2S, CO2, methanol, TEG, amine (common in oil & gas)

## 1.5 units.py (592L), validation.py (555L), constants.py (409L)
STATUS: Good. Minor improvements.
IMPROVE:
- Add concentration units: ppm, mg/m³, vol% — conversion is critical for LEL/IDLH

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: SOURCE TERM (1,144L orifice + 824L pipe + 661L two_phase + 1,128L vessel)
# ══════════════════════════════════════════════════════════════════════════════

## 2.1 orifice.py (1,145L)
CURRENT: Gas choked/subsonic, liquid Bernoulli, two-phase Omega, auto-dispatch
PROBLEMS:
- Fullbore liquid rate 1000% off vs PHAST — pipe diameter capping exists but needs data
- No connection to vessel_depressur.py for time-varying rate
IMPROVE:
- Add pipe diameter field per ISO section (from P&ID data)
- Add time-varying rate output: mdot(t) = f(P(t)) from vessel blowdown
- Add Froude number check for liquid releases (determines jet vs pool)

## 2.2 vessel_depressur.py (1,128L)
CURRENT: Full blowdown ODE solver (API 521, Haque)
PROBLEMS:
- EXISTS but NOT CONNECTED to QRA workflow
- QRA comparison used static rates because vessel blowdown isn't called from QRA
IMPROVE:
- Add function: "given inventory + hole size → time-averaged rate over duration"
- Connect to QRA: replace static rate with blowdown-derived time-averaged rate
- This FIXES the liquid fullbore problem (currently +1000% vs PHAST)

## 2.3 pipe_flow.py (824L)
PROBLEMS:
- Has pipe flow model but not connected to fullbore scenario
IMPROVE:
- For fullbore releases, model pipe flow (not just orifice) — pipe friction limits rate
- This is why PHAST gets 80 kg/s for ISO 2L while we get 1120 kg/s

## 2.4 pool_evaporation.py (581L)
STATUS: Basic pool evaporation exists.
IMPROVE:
- Add spreading model (gravity-driven pool spread on water/land)
- Connect to pool_fire.py for seamless pool fire scenario

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: DISPERSION (2,168L gaussian_plume + 747L puff + 689L dense_gas)
# ══════════════════════════════════════════════════════════════════════════════

## 3.1 gaussian_plume.py (2,168L)
CURRENT: Plume, jet-enhanced sigma, flash fire distance, binary search
PERFORMANCE: Flash fire -10% avg vs PHAST ✅
PROBLEMS:
- No building wake / terrain effect on sigma
- No time-varying source integration
IMPROVE:
- Add building wake factor (Huber-Snyder method) for congested areas
- Add continuous-to-puff transition based on release duration
- Add terrain roughness parameter (z0) — currently hardcoded as "rural"

## 3.2 gaussian_puff.py (747L)
PROBLEMS:
- Exists but NOT used by QRA workflow
- QRA comparison uses inline puff code in the comparison script
IMPROVE:
- Connect to QRA: auto-select puff vs plume based on duration
- Add virtual source correction for jet momentum

## 3.3 dense_gas.py (689L)
CURRENT: SLAB-based gravity spreading + air entrainment + passive transition
PROBLEMS:
- EXISTS but NOT connected to QRA workflow
- No H2S concentration calculation
- No validation against PHAST UDM
IMPROVE:
- Add H2S-specific parameters (molecular weight 34, density ratio ~1.2)
- Connect to toxic_dose.py for H2S fatality calculation
- Validate against PHAST H2S dispersion contours from FERA doc

## 3.4 dispersion_dispatcher.py (553L)
PROBLEMS:
- Routes to Gaussian/dense/puff based on density ratio but:
  - Not connected to source term output
  - Not connected to QRA workflow
IMPROVE:
- Make dispatcher the SINGLE ENTRY POINT for all dispersion
- Auto-select model based on: release rate, duration, density ratio, distance
- Return concentration at any (x, y, z) for vulnerability calculation

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: FIRE (1,248L jet_fire + 933L pool_fire + 622L flash_fire + 718L BLEVE)
# ══════════════════════════════════════════════════════════════════════════════

## 4.1 jet_fire.py (1,248L)
CURRENT: Multi-point source, Chamberlain flame length, refined transmissivity
PERFORMANCE: 4.73 kW/m² +4%, 6.3 kW/m² -3%, 12.5 kW/m² -20%, 37.5 kW/m² -51%
PROBLEMS:
- 37.5 kW/m² still -51% — point source fundamentally limited at near-field
- Pool fire already has solid flame view factor! (view_factor_cylinder_vertical)
IMPROVE:
- COPY solid flame view factor from pool_fire.py into jet_fire.py
- Use tilted cylinder view factor for each flame segment
- This should fix 37.5 kW/m² to within -20%

## 4.2 pool_fire.py (933L)
CURRENT: Has solid flame model with view factors! Thomas correlation, AGA tilt
PROBLEMS:
- burning_rate_default uses simple correlation — needs per-substance data
- No pool spreading model connected
IMPROVE:
- Add substance-specific burning rates (Mudan database)
- Connect pool_evaporation.py for automatic pool size calculation
- Add multi-point thermal radiation (like jet_fire)

## 4.3 flash_fire.py (622L)
CURRENT: Basic flash fire distance calculation
PROBLEMS:
- Not connected to dispersion output
- No fatality model (just distance)
IMPROVE:
- Connect to gaussian_plume output (50% LFL contour)
- Add fatality probability inside flash fire zone (0.85-0.95 typical)

## 4.4 bleve.py (718L)
STATUS: Has fireball diameter, duration, view factor, fragment throw
IMPROVE:
- Add vessel rupture frequency from failure_frequency.py
- Connect to domino analysis (BLEVE as escalation trigger)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: EXPLOSION (652L TNO + 701L TNT + 735L BST)
# ══════════════════════════════════════════════════════════════════════════════

## 5.1 All three methods exist
PROBLEMS:
- Not connected to QRA — QRA uses inline simplified TNT fallback
- BST method needs flame speed which requires congestion assessment
IMPROVE:
- Connect TNO multi-energy to QRA (it's already there!)
- Add congestion level selector (low/medium/high → BST flame speed)
- Add overpressure vs distance curves (not just single point)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: VULNERABILITY (481L probit + 350L shelter + 563L toxic + 518L calc)
# ══════════════════════════════════════════════════════════════════════════════

## 6.1 probit.py (481L)
CURRENT: Eisenberg thermal, overpressure probits
PROBLEMS:
- Only basic probits — QRA doc mentions threshold AND probit methods
- Missing: lung hemorrhage, whole-body displacement, eardrum rupture
IMPROVE:
- Add TNO probit functions (17 models from Green Book)
- Add building collapse probit
- Add thermal dose probit (not just threshold)

## 6.2 toxic_dose.py (563L)
CURRENT: Toxic load (C^n·t), ERPG/AEGL database for 20+ substances
PROBLEMS:
- H2S IN THE DATABASE but NOT connected to dispersion/dense_gas
- No probit for H2S fatality
IMPROVE:
- Add H2S probit function (TNO: Y = -31.42 + 3.008 × ln(C^1.9 × t))
- Connect dense_gas → toxic_dose → probit for H2S scenarios
- This is the #1 gap: H2S is the dominant risk in NKT QRA

## 6.3 shelter_factor.py (350L)
CURRENT: Building infiltration model, indoor concentration decay
PROBLEMS:
- EXISTS but NOT used in QRA comparison (hardcoded shelter factors)
IMPROVE:
- Connect to QRA: auto-calculate shelter factor per location type
- Add blast resistance categories (control room = blast-rated, etc.)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7: QRA (1,363L domino + 771L event_tree + 702L freq + 749L ignition)
# ══════════════════════════════════════════════════════════════════════════════

## 7.1 failure_frequency.py (702L)
CURRENT: HSE/OGP database, component types, leak sizes, modifiers
PROBLEMS:
- EXISTS but QRA comparison used HARDCODED frequencies!
- Not connected to the main QRA workflow
IMPROVE:
- Connect to QRA panel: user selects equipment type → auto-populate frequency
- Add isolatable section frequency calculation (sum of all components)
- Add modifier system: inspection quality, age, corrosion

## 7.2 ignition_prob.py (749L)
CURRENT: Cox/Lees, TNO Purple Book, HSE UK, API 581 models
PROBLEMS:
- EXISTS but QRA comparison used hardcoded ignition probabilities!
IMPROVE:
- Connect to QRA: auto-calculate ignition probability per scenario
- Select model based on substance category and release conditions

## 7.3 event_tree.py (771L)
CURRENT: Full event tree with branching, scenarios, probability calculation
PROBLEMS:
- EXISTS but NOT connected to QRA workflow!
- This is the missing link: release → event tree → consequence → LSIR
IMPROVE:
- AUTO-GENERATE event tree from: release scenario + ignition model + consequence models
- Each branch: immediate ignition → jet/pool fire, delayed ignition → flash fire/VCE
- Output: list of (frequency, consequence_type, distance) pairs

## 7.4 individual_risk.py (544L) + societal_risk.py (695L)
CURRENT: LSIR grid, IRPA, PLL, FN curve
PROBLEMS:
- LSIR calculation exists but uses hardcoded consequence distances
- Not connected to actual consequence models
IMPROVE:
- Connect: event_tree → consequence_model → LSIR_grid → IRPA → PLL → FN curve
- Add LSIR contour plotting (risk isopleth map)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8: METEOROLOGY (578L indonesia + 500L met + 530L stability + 792L weather + 614L wind_rose)
# ══════════════════════════════════════════════════════════════════════════════

## 8.1 wind_rose.py (614L) — FULL wind rose implementation
PROBLEMS:
- EXISTS but QRA only uses 2 weather scenarios (1.35C and 5.5D)
- Not connected to QRA weighting
IMPROVE:
- QRA should iterate over ALL wind rose bins, weighted by probability
- This is how SAFETI works: 12 directions × 6 speed classes × 6 stability = 432 scenarios

## 8.2 indonesia_locations.py (578L)
CURRENT: Weather data for Indonesian cities
PROBLEMS:
- Blora/CPP Gundih location data may be missing
IMPROVE:
- Add NKT-01TW location with site-specific weather data from FERA doc

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 9: TERRAIN (789L DEM + 643L LOS + 497L obstacle)
# ══════════════════════════════════════════════════════════════════════════════

## 9.1 los_engine.py (643L) — Line of Sight with obstacle shadow zones
## 9.2 dem_loader.py (789L) — Digital Elevation Model
STATUS: FULLY IMPLEMENTED but not used in QRA
IMPROVE:
- Use LOS for thermal radiation shadow zones (buildings block radiation)
- Use DEM for terrain-aware dispersion (not flat terrain assumption)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 10: REPORT (819L PDF + 437L Excel + 485L GIS + 457L image + 254L text)
# ══════════════════════════════════════════════════════════════════════════════

## 10.1 pdf_generator.py (819L)
PROBLEMS:
- Generates consequence report but NOT QRA report
IMPROVE:
- Add QRA report template: LSIR table, IRPA table, PLL table, FN curve, ALARP chart
- Add compliance statement (all workers in ALARP / intolerable)

## 10.2 gis_export.py (485L)
STATUS: Has GeoJSON/KML export for contours
IMPROVE:
- Add LSIR contour export (risk isopleths as GeoJSON)
- Add H2S dispersion contour export

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 11: ANALYSIS (532L batch + 814L monte_carlo + 471L sensitivity)
# ══════════════════════════════════════════════════════════════════════════════

## 11.1 batch_runner.py (532L)
STATUS: Exists but not connected to QRA
IMPROVE:
- Run 100+ scenarios (7 ISO × 4 holes × 2 weather = 56 base scenarios)
- Auto-tabulate IRPA/PLL per worker

## 11.2 monte_carlo.py (814L, 59 functions!)
STATUS: FULL Monte Carlo implementation
IMPROVE:
- Apply to QRA: sample frequency × consequence probability × vulnerability
- Output: confidence interval on IRPA and PLL

## 11.3 sensitivity.py (471L)
STATUS: Tornado chart + parameter variation
IMPROVE:
- Key sensitivities: leak frequency, ignition probability, wind speed, occupancy
- Show which parameter drives risk most

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 12: UI (30 files, ~15,000L)
# ══════════════════════════════════════════════════════════════════════════════

PROBLEMS:
- All individual panels exist (source_term, dispersion, fire, explosion, vulnerability, QRA, domino)
- BUT panels work IN ISOLATION — no end-to-end workflow
IMPROVE:
- Add workflow chain: source_term → dispersion → fire/explosion → vulnerability → QRA
- Add "Run Full QRA" button that chains all models
- Add scenario manager: define ISO sections + hole sizes + workers → auto-run
- Add site layout import (DXF/SVG) for plant visualization
- Add risk contour overlay on site layout

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY: THE BIGGEST ISSUE IS NOT MISSING FEATURES — IT'S DISCONNECTION
# ══════════════════════════════════════════════════════════════════════════════

Rekarisk already has 80% of what's needed for a full QRA:
- ✅ Failure frequency database (702L)
- ✅ Ignition probability models (749L)
- ✅ Event tree analysis (771L)
- ✅ Vessel blowdown (1,128L)
- ✅ Dense gas dispersion (689L)
- ✅ Toxic dose/probit (563L + 481L)
- ✅ Shelter factor (350L)
- ✅ Wind rose (614L)
- ✅ Monte Carlo (814L)
- ✅ Terrain/LOS (1,429L)
- ✅ GIS export (485L)
- ✅ LSIR/IRPA/FN calculation (1,239L)

BUT NONE OF THESE ARE CONNECTED TO EACH OTHER!

The #1 improvement is:
→ BUILD THE END-TO-END QRA PIPELINE that connects all existing modules
