# Rekarisk — Consequence Analysis Software
## Design Plan (v3 — Final)
### Feature-Complete untuk Setara Phast & SAFETI

---

## 0. Software Identity

| | |
|---|---|
| **Name** | **Rekarisk** |
| **Tagline** | *Consequence & Risk Analysis for Safety Engineers* |
| **Version** | 1.0.0-dev |
| **Codename** | "Cikal" (first milestone) |
| **Target User** | Process Safety Engineer, HSE Engineer, QRA Practitioner |
| **Platform** | Desktop (Windows, Linux, macOS) |
| **License** | Open Source (TBD: GPLv3 or MIT) |
| **Language** | English (UI); Python 3.11+ (code) |
| **Standards** | Kepmen LH, API (RP 521, RP 752, RP 753), CCPS, TNO Yellow Book, HSE UK |
| **Input** | Manual Entry |
| **Output** | Contour Maps, Tables, PDF Report, Excel, GIS Overlay (GeoJSON/Shapefile/KML) |

---

## 1. Tech Stack (Expanded)

| Layer | Technology | Rationale |
|---|---|---|
| GUI | PyQt6 / PySide6 | Mature desktop framework, native look, signal-slot |
| Computation | NumPy, SciPy | Integration, interpolation, root-finding, optimization |
| Plotting 2D | Matplotlib + Cartopy | Contour, GIS-aware, publication-grade |
| Plotting 3D | Mayavi / PyVista / plotly | 3D plume, terrain surface, isosurfaces |
| GIS Backend | Shapely, Fiona, PyProj, Rasterio | Geometry ops, projections, DEM import, GeoJSON/Shapefile |
| Equation of State | CoolProp / Cantera (optional) | Peng-Robinson, SRK for VLE, phase envelope |
| Fluid Properties | DIPPR correlations (implemented inline) | Temp-dependent vapor pressure, density, viscosity |
| PDF Report | ReportLab | PDF native, tables, embedded images |
| Excel Export | openpyxl | Multi-sheet workbooks |
| Substance DB | SQLite + JSON | Fast lookup, user-extensible |
| Auto-Updater | PyUpdater / custom + GitHub API | Check, download, verify, install |
| Packaging | PyInstaller | Single .exe / .app distributable |
| Testing | Pytest | Unit, regression, benchmark validation |
| Monte Carlo | Custom + SciPy stats | Uncertainty & sensitivity analysis |

---

## 2. Arsitektur Sistem (Expanded — 10 Modul Utama)

```
consequence-analysis/
├── src/
│   ├── models/                          # 🔬 Pure computation engine
│   │   │
│   │   ├── source_term/                 # ═══ NEW: Discharge Engine ═══
│   │   │   ├── orifice.py               # Liquid/gas/vapor leak through hole
│   │   │   ├── pipe_flow.py             # Pipe discharge + pressure drop
│   │   │   ├── vessel_depressur.py      # Time-varying vessel blowdown
│   │   │   ├── two_phase.py             # Two-phase flow (HEM, slip models)
│   │   │   ├── relief_valve.py          # PSV / RD discharge
│   │   │   ├── pool_evaporation.py      # Liquid pool spreading + evaporation
│   │   │   ├── rainout.py               # Droplet formation & rainout
│   │   │   └── aerosol.py               # Aerosol entrainment fraction
│   │   │
│   │   ├── dispersion/
│   │   │   ├── gaussian_plume.py        # Continuous release (steady-state)
│   │   │   ├── gaussian_puff.py         # Instantaneous release (time-varying)
│   │   │   ├── dense_gas.py             # SLAB simplified (gravity spread + transition)
│   │   │   ├── building_wake.py         # ═══ NEW: Cavity zone, wake entrainment ═══
│   │   │   ├── lift_off.py              # ═══ NEW: Dense→buoyant transition ═══
│   │   │   ├── indoor_outdoor.py        # ═══ NEW: Infiltration, HVAC ═══
│   │   │   └── meteorology.py           # Wind profile, Pasquill-Gifford stability
│   │   │
│   │   ├── fire/
│   │   │   ├── pool_fire.py             # Radiant heat flux (Mudan, Shokri-Beyler)
│   │   │   ├── jet_fire.py              # API RP 521
│   │   │   ├── bleve.py                 # Fireball (HSE, CCPS)
│   │   │   ├── flash_fire.py            # LFL envelope
│   │   │   ├── los_thermal.py           # ═══ NEW: Line-of-sight blocking ═══
│   │   │   └── combined_fire.py         # Multiple fires interaction
│   │   │
│   │   ├── explosion/
│   │   │   ├── tnt_equivalency.py       # Kingery-Bulmash
│   │   │   ├── tno_multi_energy.py      # TNO blast curves
│   │   │   ├── baker_strehlow.py        # BST method
│   │   │   ├── confinement.py           # ═══ NEW: Auto confinement class ═══
│   │   │   └── blast_obstacle.py        # ═══ NEW: Obstacle attenuation ═══
│   │   │
│   │   ├── qra/                         # ═══ NEW: Full QRA Module ═══
│   │   │   ├── failure_frequency.py     # Generic failure rates (HSE, OGP, OREDA)
│   │   │   ├── event_tree.py            # Event tree analysis + branching
│   │   │   ├── ignition_prob.py         # Immediate vs delayed ignition
│   │   │   ├── explosion_prob.py        # Probability of explosion given release
│   │   │   ├── individual_risk.py       # IRPA calculation
│   │   │   ├── societal_risk.py         # FN curve + criterion
│   │   │   ├── risk_matrix.py           # ISO 17776 / API risk ranking
│   │   │   ├── alarp.py                 # ALARP demo + CBA
│   │   │   └── population.py            # Day/night, indoor/outdoor fraction
│   │   │
│   │   └── vulnerability/
│   │       ├── probit.py                # Thermal, overpressure, toxic probit
│   │       ├── toxic_dose.py            # ═══ NEW: Cⁿ·t toxic load ═══
│   │       ├── shelter_factor.py        # ═══ NEW: Indoor protection ═══
│   │       └── evacuation.py            # ═══ NEW: Evacuation time window ═══
│   │
│   ├── core/
│   │   ├── substance.py                 # Material dataclass + property lookups
│   │   ├── substance_db.py              # SQLite/JSON database loader
│   │   ├── eos.py                       # ═══ NEW: Peng-Robinson, SRK, PR-BM ═══
│   │   ├── dippr.py                     # ═══ NEW: DIPPR 100-series correlations ═══
│   │   ├── phase_envelope.py            # ═══ NEW: Phase diagram calc ═══
│   │   ├── hydrate.py                   # ═══ NEW: Hydrate formation check ═══
│   │   ├── units.py                     # SI ↔ Imperial ↔ field units
│   │   ├── constants.py                 # Physical constants, regulatory thresholds
│   │   └── validation.py                # Input validation & sanity checks
│   │
│   ├── meteorology/                     # ═══ NEW: Advanced Weather ═══
│   │   ├── wind_rose.py                 # Wind rose probability table
│   │   ├── weather_jpd.py               # Joint probability distribution
│   │   ├── diurnal.py                   # Day/night PG stability shifts
│   │   └── weather_loader.py            # Import from CSV, met data formats
│   │
│   ├── terrain/                         # ═══ NEW: 3D & Terrain ═══
│   │   ├── dem_loader.py                # SRTM, GeoTIFF import
│   │   ├── obstacle.py                  # Building/obstacle modeling
│   │   ├── los_engine.py                # Line-of-sight calculator
│   │   └── terrain_viewer.py            # 3D terrain + plume overlay
│   │
│   ├── analysis/                        # ═══ NEW: Advanced Analysis ═══
│   │   ├── batch_runner.py              # Run multiple cases in sequence
│   │   ├── sensitivity.py               # Tornado chart, Sobol indices
│   │   ├── monte_carlo.py               # Monte Carlo uncertainty propagation
│   │   ├── worst_case.py                # Worst-case identification
│   │   └── audit_trail.py               # Version log, change tracking
│   │
│   ├── visualization/
│   │   ├── contour_2d.py                # 2D isopleth
│   │   ├── contour_3d.py                # ═══ NEW: 3D isosurface ═══
│   │   ├── contour_gis.py               # GeoJSON/Shapefile output
│   │   ├── fn_chart.py                  # F-N curve (log-log)
│   │   ├── risk_contour.py              # ═══ NEW: Individual risk contour ═══
│   │   ├── wind_rose_plot.py            # ═══ NEW: Wind rose chart ═══
│   │   ├── tornado_plot.py              # ═══ NEW: Tornado chart ═══
│   │   ├── cross_section.py             # Plume cross-section
│   │   └── footprint.py                 # Footprint ellipse
│   │
│   ├── report/
│   │   ├── pdf_generator.py             # Full PDF report
│   │   ├── excel_export.py              # Multi-sheet Excel
│   │   ├── text_export.py               # CSV/TXT export
│   │   └── templates/                   # Layout templates
│   │
│   └── ui/
│       ├── main_window.py               # Main window + project manager
│       ├── menu_bar.py                  # File, Edit, View, Tools, Help
│       ├── project_panel.py             # Project tree navigator
│       ├── source_term_panel.py         # ═══ NEW: Discharge input form ═══
│       ├── dispersion_panel.py          # Dispersion input form
│       ├── fire_panel.py                # Fire input form
│       ├── explosion_panel.py           # Explosion input form
│       ├── qra_panel.py                 # ═══ NEW: QRA input ═══
│       ├── weather_dialog.py            # Advanced weather input
│       ├── terrain_dialog.py            # ═══ NEW: DEM + obstacle input ═══
│       ├── substance_selector.py        # Searchable substance picker
│       ├── mixture_editor.py            # Mixture composition editor
│       ├── batch_dialog.py              # ═══ NEW: Batch run config ═══
│       ├── sensitivity_dialog.py        # ═══ NEW: Sensitivity config ═══
│       ├── monte_carlo_dialog.py        # ═══ NEW: Monte Carlo config ═══
│       ├── results_viewer.py            # Tabbed result viewer
│       ├── contour_canvas.py            # Matplotlib embedded canvas
│       ├── contour_canvas_3d.py         # ═══ NEW: 3D viewer ═══
│       ├── table_view.py                # QTableView with export
│       ├── gis_export_dialog.py         # GIS export options
│       └── audit_viewer.py              # ═══ NEW: Audit trail viewer ═══
│
├── data/
│   ├── substances.json                  # ~150 hazardous substances
│   ├── diprr_params.json                # ═══ NEW: DIPPR correlation coefficients ═══
│   ├── probit_constants.json            # Probit coefficients (TNO, HSE, CCPS)
│   ├── pasquill_coefficients.json       # Sigma-y, sigma-z coefficients
│   ├── failure_rates.json               # ═══ NEW: Generic failure frequencies ═══
│   ├── ignition_probs.json              # ═══ NEW: Ignition probability data ═══
│   ├── blast_curves/                    # TNO & BST blast curve tables
│   └── risk_criteria.json               # ═══ NEW: Risk acceptability thresholds ═══
│
├── tests/
│   ├── test_source_term/
│   ├── test_dispersion/
│   ├── test_fire/
│   ├── test_explosion/
│   ├── test_qra/
│   ├── test_eos/
│   └── benchmarks/                      # Published validation cases
│
├── docs/
│   ├── USER_MANUAL.md
│   ├── METHODOLOGY.md
│   └── VALIDATION.md
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. Modul Detail

### 3A. SOURCE TERM / DISCHARGE ENGINE (Modul Baru — Fondasi)

Ini adalah modul paling kritis yang belum ada di plan awal. Sebelum menghitung dispersi/kebakaran/ledakan, engineer harus tahu **berapa laju release** — dan ini adalah bagian tersulit.

**3A.1 Orifice Discharge (Leak through hole)**

```
Sub-models:
  - Liquid release (Bernoulli):  Q = C_d·A·ρ·√[2(P₁-P₂)/ρ + 2gH]
  - Gas/vapor release (choked/unchoked):
    - P_choked = P₁·(2/(γ+1))^(γ/(γ-1))
    - If P₂ < P_choked → choked (sonic) flow
    - If P₂ ≥ P_choked → subsonic

Input:
  - Vessel/storage conditions (T, P)
  - Hole diameter (mm) — small/medium/large preset
  - Fluid phase: liquid / gas / two-phase
  - Discharge coefficient (C_d): 0.6-1.0
  - Elevation head, duration

Output:
  - Initial release rate (kg/s)
  - Phase at orifice
  - Total mass released over duration
  - Velocity at orifice
  - Expansion to atmospheric pressure
```

**3A.2 Pipe Flow / Pipe Rupture**

```
Model: Darcy-Weisbach + friction factor
  - Full-bore rupture: guillotine break
  - Pipe leak: hole-in-pipe
  - Long pipeline: account for friction over distance
  - Two-phase pipe flow: Beggs-Brill or Lockhart-Martinelli

Input:
  - Pipe ID, length, roughness
  - Upstream P, T
  - Rupture type (full bore / leak)
```

**3A.3 Vessel Depressurization (Blowdown)**

```
Model: Time-varying vessel conditions
  - Mass & energy balance over time
  - Vessel cooling (Joule-Thomson)
  - Liquid level changes
  - Heat transfer from vessel wall & ambient
  - Two scenarios:
    a) Gas-only blowdown
    b) Liquid + gas two-phase blowdown
  - Numerical integration (odeint)

Output:
  - Release rate vs time curve
  - Vessel P, T vs time
  - Total mass released
```

**3A.4 Two-Phase Flow**

```
When: Liquid flashes as it exits → liquid + vapor mixture

Models:
  - HEM (Homogeneous Equilibrium Model) — simplest
  - Slip models (Fauske, Moody) — for non-equilibrium
  - Omega method (API 520, Leung)

Output:
  - Two-phase mass flux (kg/s·m²)
  - Vapor mass fraction at exit
  - Jet velocity
  - Expansion behavior
```

**3A.5 Relief Valve / PSV**

```
Model: API 520 Part I
  - Critical & subcritical flow
  - Gas, vapor, steam, liquid
  - Two-phase relief (Omega method)
  - Backpressure effects
```

**3A.6 Liquid Pool Spreading & Evaporation**

```
After spill:
  1. Pool spreading on ground (gravity-inertia → gravity-viscous)
  2. Pool evaporation → vapor feed to dispersion

Input:
  - Spill mass / rate
  - Surface type (concrete, soil, water)
  - Ambient T, wind speed
  - Bund/dike containment (optional)

Models:
  - Spreading: Shaw-Briscoe, Webber
  - Evaporation: Mackay-Matsugu (mass transfer)
  - Boiling pool: if T_ambient > T_boil
  - Cryogenic pools: LNG/LIN on water → rapid phase transition

Output:
  - Pool radius vs time
  - Evaporation rate vs time
  - Pool lifetime
  - Vapor generation rate → feed to dispersion model
```

**3A.7 Rainout & Aerosol**

```
Two-phase jet release:
  - Some liquid droplets are small enough → carried as aerosol
  - Some droplets are large → rain out → form liquid pool

Model: droplet size distribution (Rosin-Rammler)
  - Weber number criterion for droplet breakup
  - Critical droplet diameter for rainout
  - Aerosol fraction vs rainout fraction

Output:
  - Aerosol mass fraction (entrained in cloud)
  - Rainout mass fraction (forms pool)
  - Effective release rate to dispersion model
```

---

### 3B. DISPERSION (Diperluas)

**3B.1 Gaussian Plume (Continuous Release)**

```
Concentration:
  C(x,y,z) = (Q/(2π·u·σy·σz)) · exp(-y²/2σy²)
            · [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)]

Features:
  - Buoyant plume rise (Briggs)
  - Dry deposition velocity
  - Wet deposition (washout coefficient)
  - Chemical decay (1st-order)
  - Ground reflection
  - Crosswind-averaged mode for societal risk
```

**3B.2 Gaussian Puff (Instantaneous Release)**

```
Time-varying 3D concentration:
  - Puff centroid: advection by wind
  - Puff growth: σ(t) from travel time
  - Multiple puffs for finite-duration release
  - Puff splitting for wind shear
```

**3B.3 Dense Gas Dispersion (SLAB-based)**

```
Phases:
  1. Gravity spreading (slumping)
  2. Air entrainment (top + edge)
  3. Heating from ground
  4. Transition to passive Gaussian
     (when density ratio < 1.01 or Richardson < critical)

Applicable substances:
  Cl₂, NH₃, LNG, LPG, SO₂, H₂S, ethylene oxide, etc.
  Criterion: ρ_cloud/ρ_air > 1.1
```

**3B.4 Building Wake Dispersion (NEW)**

```
After release near buildings:
  - Cavity zone: recirculation behind building
    Length ≈ 3 × building width
  - Wake entrainment: enhanced mixing
  - Downwash: plume pulled toward ground
  - Source in cavity → well-mixed concentration in cavity

Model: ASHRAE / SCREEN3 building downwash
  - Building dimensions (H, W, L)
  - Stack height vs building height
  - Wind direction relative to building face
```

**3B.5 Dense Gas Lift-off (NEW)**

```
When: Dense gas heated by ground → becomes buoyant → rises

Model:
  - Continuous tracking of cloud density
  - Transition when ρ_cloud < ρ_air
  - Modified Briggs plume rise for lifted cloud
  - Important for: LNG vapor clouds (cold but CH₄ is buoyant)
```

**3B.6 Indoor/Outdoor (NEW)**

```
Indoor concentration from outdoor release:
  - Air exchange rate (ACH) — building ventilation
  - Infiltration through cracks
  - HVAC fresh air intake
  - Time-dependent indoor concentration
  - Shelter-in-place effectiveness
```

---

### 3C. FIRE (Diperluas)

**3C.1 Pool Fire**

```
Model: solid flame cylinder / tilted cylinder (wind effect)
  - Burning rate: m" = m"_∞ · (1 - exp(-κβD))
  - Flame height: H/D correlation (Thomas, Heskestad)
  - Flame tilt: from wind speed / Fr number
  - Surface Emissive Power: SEP
  - View factor: analytical (cylinder) or numerical (flame surface discretization)
  - Atmospheric transmissivity
  - Smoke obscuration factor

Thresholds (from API 752):
  37.5 kW/m² — equipment damage / auto-ignition
  12.5 kW/m² — piloted ignition of wood
   5.0 kW/m² — 2nd degree burn (40s exposure)
   1.6 kW/m² — solar radiation equivalent (no hazard)
```

**3C.2 Jet Fire**

```
Model: API RP 521
  - Cone/flare-shaped flame
  - Sonic vs subsonic jet
  - Multi-point source radiation
  - Flame length by API correlation
  - Radiant fraction: 0.15–0.40 depending on gas
```

**3C.3 BLEVE / Fireball**

```
Models (selectable):
  - HSE method
  - CCPS method
  - TNO method

Fireball properties:
  - Diameter: D = 5.8 · M^(1/3) (HSE/CCPS)
  - Duration: t = 0.45·M^(1/3) or 2.6·M^(1/6) depending on mass
  - Height: 0.75 · D above ground
  - Surface emissive power
  - Transient radiation (time integration for dose)
```

**3C.4 Flash Fire**

```
Method: dispersion model → LFL isopleth
  - ½ LFL contour = conservative hazard zone
  - LFL contour = flame envelope
  - Assumption: inside LFL = 100% fatality
  - No thermal radiation calc (transient flame, low duration)
```

**3C.5 Line-of-Sight Blocking (NEW)**

```
For thermal radiation:
  - Terrain/buildings between fire and receptor
  - Ray casting algorithm from fire surface to receptor point
  - Partially blocked → reduced view factor
  - Fully blocked → zero heat flux
  
Important for: occupied buildings behind blast walls, terrain ridges
```

---

### 3D. EXPLOSION

**3D.1 TNT Equivalency**

```
M_TNT = η · M · ΔHc / 4680

Kingery-Bulmash polynomial:
  log₁₀(P_s) = Σ cᵢ·(a + b·log₁₀(Z))ⁱ           for i=0..n
  where P_s = overpressure (bar), Z = R / M_TNT^(1/3)

Efficiency factors (η):
  - Confined: 0.02–0.10
  - Partially confined: 0.01–0.05
  - Unconfined: 0.001–0.02
  
+ Impulse calculation
+ Reflected pressure for near-field
```

**3D.2 TNO Multi-Energy**

```
Procedure:
  1. Assess confinement / obstruction → blast strength (1-10)
     Strength 1:  unconfined (deflagration)
     Strength 7-10: highly confined (detonation)
  2. Combustion energy: E = M_confined · ΔHc
  3. Sachs-scaled distance: R̄ = R / (E/P_atm)^(1/3)
  4. Interpolate digitized TNO blast curves
  5. P_s, positive phase duration, impulse per distance

Strength selection is CRITICAL — wrong choice = wrong answer.
Provide guidance table based on:
  - Congestion level (low/medium/high)
  - Confinement level (unconfined/partially/fully)
  - Ignition energy (weak/strong)
```

**3D.3 Baker-Strehlow-Tang (BST)**

```
Better for congested process plants:

Step 1: Determine flame Mach number (M_f)
  Based on:
  - Congestion: low / medium / high
  - Confinement: 1D, 2D, 3D
  - Fuel reactivity: low / medium / high

Step 2: Look up blast curve for M_f
  - M_f 0.2:   deflagration in open
  - M_f 0.5-1.0: congested deflagration
  - M_f 2.0+:    transition to detonation
  - M_f 5.2:     stable detonation (CJ)

Output: P_s & impulse vs scaled distance
```

**3D.4 Confinement Assessment (NEW)**

```
Auto-assessment tool:
  - User draws bounding polygon on obstacle layout
  - Calculate: V_cloud / V_confined
  - Count obstacles in confined region
  - Congestion: VBR (Volume Blockage Ratio)
  - Suggest blast strength based on VBR + confinement
```

**3D.5 Obstacle Effects (NEW)**

```
Blast wave interaction with obstacles:
  - Reflection: up to 2-8× pressure at building face
  - Diffraction: around building edges
  - Drag loading: F = C_d · A · P_dyn
  - Building damage assessment per ASCE / CCPS
```

---

### 3E. QRA FRAMEWORK (Modul Baru)

Ini adalah salah satu fitur utama SAFETI — menghubungkan konsekuensi dengan frekuensi untuk menghasilkan *risk*.

**3E.1 Failure Frequency Database**

```
Generic failure rates (per year) — sources:
  - UK HSE Hydrocarbon Release Database
  - OGP Risk Assessment Data Directory
  - OREDA (Offshore Reliability Data)
  - Purple Book (Netherlands)
  - API RP 581 (RBI)

Equipment types:
  - Pressure vessels: 1×10⁻⁵ to 5×10⁻⁵ /yr (catastrophic)
  - Piping (per meter): 1×10⁻⁶ to 5×10⁻⁵ /yr (full-bore)
  - Pumps: 1×10⁻⁴ /yr (catastrophic)
  - Storage tanks: 1×10⁻⁵ to 5×10⁻⁵ /yr
  - Flanges/valves: 1×10⁻⁶ to 1×10⁻⁵ /yr (leak)

Hole size distribution per equipment:
  - Small: 5-10 mm
  - Medium: 10-50 mm
  - Large: 50-150 mm
  - Rupture: >150 mm (full-bore)
```

**3E.2 Event Tree Analysis (ETA)**

```
For each release scenario:
  1. Release occurs
  2. Immediate ignition? → Yes: fire/explosion; No: dispersion
  3. Delayed ignition? → Yes: VCE or flash fire; No: safe dispersion
  4. Explosion? → Yes: blast; No: flash fire
  5. Each branch → probability × consequence

Example event tree:
  Release (f = 1×10⁻⁴/yr)
  ├─ Immediate ignition (p=0.05) → Jet fire / Pool fire (P_f=5×10⁻⁶/yr)
  ├─ No imm ign (p=0.95)
  │  ├─ Delayed ignition (p=0.30) → VCE (P_f=2.85×10⁻⁵/yr)
  │  └─ No delayed ign (p=0.70) → Safe (no consequence)
```

**3E.3 Ignition Probability**

```
Immediate ignition probability:
  - From equipment type, release rate
  - Cox, Lees, Ang model
  - Gas > 10 kg/s: p ~ 0.1-0.3
  - Liquid: much lower

Delayed ignition probability:
  - From cloud size & ignition source density
  - Offshore: high ignition density
  - Onshore rural: low ignition density
  - Onshore industrial: medium-high
```

**3E.4 Individual Risk (IRPA)**

```
Individual Risk Per Annum:
  IR(x,y) = Σ f_i · P_fatality,i(x,y)

Display:
  - IR contour: 1×10⁻⁴, 1×10⁻⁵, 1×10⁻⁶ /yr
  - ISO risk criteria overlay
  - Worst location identification

Typical criteria (UK HSE / ISO):
  - IR > 10⁻⁴/yr: intolerable for workers
  - IR 10⁻⁵ to 10⁻⁴/yr: tolerable if ALARP
  - IR < 10⁻⁶/yr: broadly acceptable
  - IR > 10⁻⁶/yr: intolerable for public
```

**3E.5 Societal Risk (F-N Curve)**

```
FN calculation:
  1. For each scenario i:
     - N_i = Σ P_fatality (sum over all grid points)
     - f_i = scenario frequency
  2. Sort by N (descending)
  3. Cumulative F = Σ f_j for all j where N_j ≥ N_k
  4. Plot: F vs N on log-log

Acceptability:
  - UK HSE criterion line
  - Netherlands (Purple Book) criterion
  - User-defined custom criterion
```

**3E.6 Risk Matrix**

```
Per ISO 17776 / API:
  - Consequence categories: 1-5 (negligible → catastrophic)
  - Likelihood categories: A-E (rare → frequent)
  - Matrix: Acceptable / ALARP / Unacceptable
  - Color-coded visualization
```

**3E.7 ALARP Demonstration**

```
For risks in the ALARP region:
  1. Identify risk reduction measures
  2. Cost-benefit analysis
  3. Disproportionality factor (usually 3-6×)
  4. Justification if cost > benefit × factor
```

**3E.8 Population Data**

```
Per geographic cell:
  - Daytime population (workers in plant, office)
  - Nighttime population (security only)
  - Indoor fraction (and building type)
  - Outdoor fraction
  - Vulnerability factor (general public vs trained workers)
```

---

### 3F. ADVANCED METEOROLOGY (Modul Baru)

**3F.1 Wind Rose — Probabilistic Weather**

```
Instead of: "Stability D, wind 5 m/s from West"
Use: Full joint probability distribution

Weather table:
  Wind direction (16 sectors) × Wind speed (bins) × Stability (A-F)
  Each cell: probability (fraction of year)

Example input:
  - CSV import from met station data
  - Manual entry: fraction per cell
  - Default: typical tropical/industrial weather

Usage:
  - Each weather case run automatically
  - Results weighted by probability
  - Risk integrated over all weather
```

**3F.2 Diurnal Variation**

```
PG stability shifts by time of day:
  - Daytime (06-18): typically A-C (unstable to neutral)
  - Nighttime (18-06): typically D-F (neutral to stable)
  - Transition periods: 1-2 hour shift

Apply different weather based on:
  - Time of release (if known: maintenance schedule)
  - Random release timing (uniform over year)
```

**3F.3 Seasonal Effects**

```
- Monsoon season: different wind patterns, higher humidity
- Snow cover: increased surface albedo, stable conditions
- Import seasonal weather tables (4 seasons or 12 months)
```

---

### 3G. 3D & TERRAIN (Modul Baru)

**3G.1 Digital Elevation Model (DEM)**

```
Import formats:
  - GeoTIFF (.tif) — SRTM, ASTER
  - HGT (SRTM raw)
  - ASCII Grid (.asc, .grd)
  - User-defined grid points

Resolution:
  - 30m (SRTM 1 arc-sec) — free global coverage
  - 90m (SRTM 3 arc-sec)
  - User custom resolution

Automatic: download SRTM tiles via API (if internet)
```

**3G.2 Obstacle / Building Modeling**

```
Representation:
  - Bounding box: length × width × height (m)
  - Position: X, Y, Z in project coordinates
  - Porosity: 0 (solid wall) to 1 (open frame)
  - Multiple obstacles per site
  
Import: CSV of building coordinates + dimensions
Manual: draw rectangles on map
```

**3G.3 Line-of-Sight Engine**

```
Ray casting for thermal radiation:
  - From fire surface (n points) to receptor
  - Check intersection with: terrain surface, building boxes
  - First hit → blocked
  - Output: fraction of rays hitting target
  - Modified view factor = F_view · (fraction unblocked)

For overpressure:
  - Building shielding: attenuation factor
  - Terrain channeling: amplification in valleys
```

**3G.4 3D Visualization**

```
3D view (Mayavi / PyVista):
  - Terrain surface (from DEM)
  - Buildings as boxes
  - 3D plume isosurface
  - Particle trajectories
  - Rotate, zoom, pan
  - Export: 3D view as image/PDF
```

---

### 3H. ADVANCED SUBSTANCE PROPERTIES (Modul Baru)

**3H.1 DIPPR Temperature-Dependent Properties**

```
DIPPR 100-series equation:
  Y = A / [B^(1 + (1 - T/C)^D)]

For 10+ properties per substance:
  - Liquid density (equation 105/106)
  - Vapor pressure (equation 101)
  - Heat of vaporization (equation 106)
  - Liquid heat capacity (equation 100)
  - Vapor heat capacity (equation 107)
  - Liquid thermal conductivity (equation 100)
  - Vapor thermal conductivity (equation 102)
  - Liquid viscosity (equation 101)
  - Vapor viscosity (equation 102)
  - Surface tension (equation 106)

Why: Release temperature ≠ boiling point
     Vessel at 60°C → different properties than at 25°C
     This is essential for Phast-level accuracy
```

**3H.2 Equation of State**

```
Cubic EoS:
  - Peng-Robinson (recommended for hydrocarbons)
  - Soave-Redlich-Kwong (SRK)
  - PR-BM (PR + Boston-Mathias alpha function)

Applications:
  - Vapor-liquid equilibrium (VLE flash)
  - Compressibility factor (Z)
  - Density of gas/vapor
  - Enthalpy departure
  - Speed of sound

Libraries: CoolProp (opensource) or implement PR directly
  - PR implementation: ~200 lines Python, well-documented
```

**3H.3 Phase Envelope**

```
P-T diagram for pure component:
  - Vapor pressure curve (DIPPR or Antoine)
  - Critical point
  - Triple point

For mixture:
  - Phase envelope (bubble + dew line)
  - At given T,P → determine phase(s)
  - Flash calculation: vapor fraction from T,P
```

**3H.4 Hydrate Formation Check**

```
For gas pipelines / gas releases:
  - Hydrate formation T at given P
  - If T_ambient < T_hydrate → hydrate risk
  - Methane hydrate: ~0°C at 25 bar
  
Relevant for:
  - LNG / natural gas releases
  - Cold gas pipeline ruptures
```

**3H.5 Water-Reactive Chemicals**

```
Special handling for:
  - TiCl₄ + H₂O → TiO₂ + 4HCl (toxic cloud)
  - SO₃ + H₂O → H₂SO₄ (acid mist)
  - Acetyl chloride → HCl generation
  - Oleum → SO₃ release

Database flag: "water_reactive" → special chemistry module
  - Stoichiometric HCl/SO₂/H₂S generation
  - Heat of reaction (exothermic → additional evaporation)
```

---

### 3I. ADVANCED TOXICOLOGY (Modul Baru)

**3I.1 Toxic Load (Dose-Based)**

```
Not just concentration — Cⁿ·t (toxic load):

  Toxic Load = ∫ [C(t)]^n dt    (from 0 to exposure_duration)

  where n (typically 1-4):
  - n=1: simple asphyxiants
  - n=1: chlorine (some models)
  - n=2: ammonia, SO₂
  - n=3: H₂S
  - n=4: phosgene

Probit (toxic): Pr = a + b·ln(Toxic Load)
  Different from concentration-based probit — more accurate
```

**3I.2 Shelter / Indoor Protection**

```
Shelter factor:
  C_indoor(t) = C_outdoor(t) · (1 - exp(-ACH·t)) / ACH

  ACH = air changes per hour
  - Sealed building: ACH 0.3-0.5
  - Normal building: ACH 1-3
  - Open structure: ACH 10+

  Integrated dose:
  - If exposure > ACH⁻¹ → indoor~outdoor (steady state)
  - Short release: indoor concentration lag → protection
  
  Evacuation: if people leave building → reduced exposure
```

**3I.3 Evacuation Time Window**

```
For emergency planning:
  1. Calculate concentration arrival time at receptor
  2. Calculate time to reach IDLH / ERPG-3
  3. Time available for evacuation = arrival - threshold time
  4. Include: alarm delay, response time, travel speed
  5. Output: "Evacuation must begin within X minutes"
```

---

### 3J. ADVANCED ANALYSIS TOOLS (Modul Baru)

**3J.1 Batch Runner**

```
Run multiple scenarios automatically:
  - All hole sizes (5mm, 25mm, 100mm, rupture)
  - All weather conditions from wind rose
  - All wind directions (8 or 16 sectors)
  - Result: matrix of hazard distances

Input:
  - Parameter ranges
  - Grid: combinations or custom list
  - Priority: run in parallel (multiprocessing)

Output:
  - Summary table: worst case per direction
  - Export all results to Excel multi-sheet
```

**3J.2 Sensitivity Analysis**

```
Tornado chart:
  1. Define base case
  2. Vary each parameter ±X% (e.g., ±20%)
  3. Record change in output (e.g., hazard distance)
  4. Rank parameters by impact
  5. Plot: horizontal bars, longest on top

Example parameters varied:
  - Hole diameter (±50%)
  - Pressure (±20%)
  - Wind speed (±30%)
  - Stability (±1 class)

Output:
  - Tornado chart
  - Sensitivity table
  - Identifies: which parameter dominates → focus QA on that
```

**3J.3 Monte Carlo Uncertainty**

```
Monte Carlo simulation:
  1. Define input distributions:
     - Normal: pressure, temperature (mean ± σ)
     - Lognormal: hole size
     - Triangular: wind speed (min, max, best)
     - Uniform: wind direction
  2. Sample N times (e.g., 10,000)
  3. Run model for each sample
  4. Output distribution:

Result statistics:
  - P50 (median) hazard distance
  - P95 (95th percentile) → conservative design
  - P99 → worst credible case
  - Histogram of output
  - CDF plot

Use: Replace "what if?" with probabilistic answer
     "There is 95% confidence that hazard zone < 350m"
```

**3J.4 Worst-Case Identification**

```
From batch / Monte Carlo results:
  - Rank all scenarios by: hazard distance / fatalities
  - Dominant scenarios (>X% of total risk)
  - Worst credible weather: F-stability, low wind
  - Worst direction: toward populated area

Output:
  - "Worst-case scenario: 100mm hole, F-stability, 2 m/s"
  - "Hazard distance: 850m (ERPG-2)"
  - "Drives 65% of total risk"
```

**3J.5 Audit Trail & Version Control**

```
Every project change logged:
  - Created by (user name)
  - Timestamp
  - What changed (input parameter, value old→new)
  - Why (user comment / reason for change)

Features:
  - Diff viewer: compare two versions
  - Rollback to previous version
  - Export audit log
  - Checkpoint before sensitivity/Monte Carlo runs

Use: Regulatory submission requires traceability
     "Who changed the hole size from 10mm to 25mm and why?"
```

---

## 4. Database Zat (Diperluas)

### 4A. Pure Substance Data

`substances.json` — target ~150 pure substances:

```json
{
  "name": "Chlorine",
  "cas": "7782-50-5",
  "formula": "Cl2",
  "mw": 70.906,
  "nbp": 239.05,
  "tc": 417.15,
  "pc": 7.991e6,
  "omega": 0.073,
  "density_liquid": 1574,
  "density_gas": 3.21,
  "heat_of_combustion": null,
  "heat_of_vaporization": 287000,
  "lfl": null, "ufl": null,
  "idlh": 10,
  "erpg1": null, "erpg2": 3.0, "erpg3": 20.0,
  "aegl1_60min": 0.5, "aegl2_60min": 2.0, "aegl3_60min": 20.0,
  "toxic_n": 2.0,
  "probit_a": -8.29, "probit_b": 0.92,
  "dippr_vp": {"A": 74.018, "B": -3666.0, "C": -7.45, "D": 7.96e-06, "E": 2.0},
  "dippr_liq_dens": {"A": 1.56, "B": 0.27, "C": 417.15, "D": 0.29},
  "water_reactive": false,
  "tags": ["toxic", "dense_gas", "highly_hazardous"]
}
```

### 4B. DIPPR Correlation Parameters

```
Separate file: dippr_params.json
  For each substance, stores A, B, C, D, E for each property
  Sources:
  - DIPPR 801 database (commercial — but published values for common fluids)
  - Yaws' Handbook of Thermodynamic Properties
  - NIST REFPROP
  - Published literature
```

### 4C. Mixture Support

```
User-defined mixtures (expanded from v1 plan):

Properties calculated:
  - MW_mix = Σ w_i · MW_i
  - P_v_mix = Σ x_i · P_v_i · γ_i  (Raoult + activity model)
  - ΔHc_mix = Σ w_i · ΔHc_i (weight average)
  - LFL_mix = 100 / Σ(y_i / LFL_i)  (Le Chatelier)
  - UFL_mix = 100 / Σ(y_i / UFL_i)
  - Density from EoS mixing rules
  - Viscosity: mixing rules (Wilke for gas, Grunberg-Nissan for liquid)

Phase behavior:
  - Flash calculation at release conditions
  - Bubble point / dew point of mixture
  - VLE using EoS + mixing rules (van der Waals one-fluid)
```

---

## 5. UI Design (Updated Layout)

```
┌──────────────────────────────────────────────────────────┐
│  Menu: File  Edit  View  Tools  Analysis  Help           │
│  File: New / Open / Save / Save As / Export / Exit       │
│  Tools: Substance DB / Mixture Editor / Options          │
│  Analysis: Batch Run / Sensitivity / Monte Carlo / Audit │
├──────────┬───────────────────────────────────────────────┤
│ PROJECT  │                                               │
│ NAVIGATOR│           RESULT VIEWER                       │
│          │  ┌──────────────────────────────────────┐    │
│ 📁 Plant │  │ [Contour 2D] [Contour 3D]            │    │
│  ├🔥 Pool│  │ [Table] [FN Curve] [Report Preview]  │    │
│  ├💨 Disp│  └──────────────────────────────────────┘    │
│  ├💥 VCE │  ┌──────────────────────────────────────┐    │
│  └📊 QRA │  │                                      │    │
│          │  │   Matplotlib / PyVista Canvas         │    │
│ Overlay: │  │   (Contour + Terrain + Buildings)     │    │
│  ☑ Case1│  │                                      │    │
│  ☑ Case2│  │                                      │    │
│  ☐ Case3│  └──────────────────────────────────────┘    │
│          │  ┌──────────────────────────────────────┐    │
│          │  │ Status: Ready | Calc time: 0.3s      │    │
│          │  └──────────────────────────────────────┘    │
├──────────┴───────────────────────────────────────────────┤
│  INPUT FORM (context-sensitive, tabbed)                   │
│  ┌─ Scenario ───┬─ Source Term ──┬─ Meteorology ────────┐│
│  │ Substance ▼  │ Vessel P: ____ │ Wind Speed: ____ m/s ││
│  │ Hole: 25mm▼  │ Vessel T: ____ │ Stability: D ▼       ││
│  │ Phase: liq▼  │ Vol: _____ m³ │ Direction:  NE ▼     ││
│  │ Duration:___ │ Elevation: ___ │ Roughness: 0.1 ▼    ││
│  └──────────────┴────────────────┴──────────────────────┘│
│  [Terrain...]  [Population...]  [Run ▶]  [Batch Run...]  │
└──────────────────────────────────────────────────────────┘
```

**Tabbed Input Form:**
1. **Scenario** — substance, hole size, phase, duration, release type
2. **Source Term** — vessel/storage conditions, pipe specs, pool specs
3. **Meteorology** — single weather or wind rose table
4. **Terrain** — DEM import, obstacle list
5. **Population** — grid-based population data (for QRA)
6. **Advanced** — deposition, decay, duration, receptor grid

**Result Viewer Tabs:**
1. **Contour 2D** — standard Matplotlib with Cartopy basemap
2. **Contour 3D** — PyVista/Mayavi with terrain
3. **Table** — hazard distances, QTableView, sortable
4. **FN Curve** — log-log plot with criteria lines (for QRA)
5. **Report Preview** — PDF preview inline

---

## 6. Standar & Regulasi

| Area | Standard | What's Implemented |
|---|---|---|
| Siting | API RP 752 | Thermal + overpressure thresholds for occupied buildings |
| Siting | API RP 753 | Toxic release siting |
| Relief | API RP 520/521 | PSV sizing, jet fire model |
| QRA | CCPS CPQRA | Framework, event tree, probit |
| QRA | ISO 17776 | Risk matrix |
| Effects | TNO Yellow Book | Pool fire, BLEVE, VCE, dispersion |
| Effects | TNO Green Book | Toxic damage, probit constants |
| Risk | UK HSE | FN criteria, IRPA |
| Risk | Purple Book (NL) | QRA methodology, failure rates |
| Environ | Kepmen LH 50/1999 | Noise thresholds |
| Environ | Kepmen LH 51/1999 | Air quality thresholds |
| Toxic | NIOSH/OSHA | IDLH, PEL |
| Toxic | EPA | AEGL-1/2/3, ERPG-1/2/3 |
| RBI | API RP 581 | Failure rates, damage factors |

---

## 7. Output (Expanded)

### 7A. Contour Maps
- 2D isopleth with terrain basemap (satellite / street / topo)
- 3D isosurface with terrain + buildings
- Multi-case overlay with per-case legend
- Scale bar, wind rose, compass rose, north arrow
- Export: PNG (300 DPI), SVG, GeoTIFF

### 7B. Tables
- Hazard distance to each threshold
- Full receptor grid data (exportable)
- Case comparison table (side-by-side)
- Batch run summary

### 7C. PDF Report (Professional)
- Cover page
- Executive summary
- Methodology (which model, which assumptions)
- Input summary table
- Source term results
- Contour figures (embedded, full page)
- Hazard zone table
- FN curve + risk matrix (for QRA)
- Sensitivity tornado chart
- Conclusion & recommendations
- Appendices: substance data, weather data, audit log
- Custom header/footer

### 7D. Excel Export
- Multi-sheet workbook:
  - Sheet 1: Input summary
  - Sheet 2: Hazard distances
  - Sheet 3: Full grid data
  - Sheet 4: FN data
  - Sheet 5: Batch comparison

### 7E. GIS Export
- GeoJSON (web mapping)
- Shapefile (QGIS/ArcGIS)
- KML (Google Earth)
- WGS84 projection (configurable to UTM)

### 7F. Text/CSV Export
- Comma/tab delimited
- Hazard summary
- Grid data

---

## 8. Development Roadmap (Full — 14 Phases)

### Phase 0: Repository & CI/CD Setup (Day 1)
- [ ] Create GitHub repository: `github.com/arienug2000/rekarisk`
- [ ] Initialize Git repo locally
- [ ] README.md with project overview
- [ ] .gitignore (Python, PyInstaller, IDE files)
- [ ] LICENSE file (TBD: GPLv3 or MIT)
- [ ] GitHub Actions CI: pytest on push/PR
- [ ] Branch protection: main branch
- [ ] First commit + push

### Phase 1: Foundation (2-3 weeks)
- [ ] Project scaffolding (directory, setup.py, requirements.txt)
- [ ] Docs scaffolding: INSTALLATION.md, USER_MANUAL.md (outline), TUTORIAL.md (outline), TEST_CASES.md (outline)
- [ ] Help system: help_topics.json skeleton, F1 context-mapping
- [ ] Core: substance dataclass + JSON database loader
- [ ] Core: DIPPR correlations (at least vapor pressure, liquid density)
- [ ] Core: units system (SI ↔ Imperial ↔ field)
- [ ] Core: physical constants + regulatory thresholds
- [ ] Core: input validation & sanity checks
- [ ] UI: Main window skeleton + menu bar
- [ ] UI: Project navigator (tree view)
- [ ] UI: Substance selector + search widget
- [ ] Project file format: New / Open / Save / Save As (*.caproj)
- [ ] Dirty flag + unsaved changes warning
- [ ] Test framework (pytest + CI)

### Phase 2: Source Term Engine (2-3 weeks)
- [ ] Orifice discharge (liquid/gas/vapor)
- [ ] Two-phase flow models (HEM, slip)
- [ ] Pipe flow / rupture model
- [ ] Vessel depressurization (blowdown)
- [ ] Relief valve (API 520)
- [ ] Liquid pool spreading & evaporation
- [ ] Rainout & aerosol fraction
- [ ] Source term input panel (UI)
- [ ] Source term result table + plots

### Phase 3: Equation of State & Advanced Props (1-2 weeks)
- [ ] Peng-Robinson EoS (pure + mixture)
- [ ] SRK EoS (optional)
- [ ] VLE flash calculation
- [ ] Phase envelope calculation
- [ ] Full DIPPR properties (10+ per substance)
- [ ] Mixture: composition input + auto-property calculation
- [ ] Hydrate formation check
- [ ] Water-reactive chemical module
- [ ] Database: expand to ~150 substances
- [ ] UI: Mixture editor dialog

### Phase 4: Meteorology Module (1 week)
- [ ] Pasquill-Gifford stability classes
- [ ] Wind profile (power law + log law)
- [ ] Surface roughness parameterization
- [ ] Wind rose table + joint probability distribution
- [ ] Diurnal variation (day/night shifts)
- [ ] Seasonal weather tables
- [ ] Weather import (CSV, met station formats)
- [ ] UI: Weather dialog + wind rose plot

### Phase 5: Dispersion (2 weeks)
- [ ] Gaussian plume (continuous)
- [ ] Gaussian puff (instantaneous, time-varying)
- [ ] Dense gas model (SLAB-based)
- [ ] Building wake / cavity zone
- [ ] Dense gas lift-off
- [ ] Indoor/outdoor infiltration
- [ ] Dry/wet deposition
- [ ] Chemical decay
- [ ] Dispersion input panel (UI)
- [ ] 2D concentration contour
- [ ] Cross-section & footprint plots
- [ ] Result table

### Phase 6: Fire (2 weeks)
- [ ] Pool fire (all correlations)
- [ ] Jet fire (API 521)
- [ ] BLEVE / fireball (HSE + CCPS)
- [ ] Flash fire (LFL isopleth from dispersion)
- [ ] Line-of-sight blocking (LOS engine)
- [ ] Multiple fire interaction
- [ ] Fire input panel (UI)
- [ ] Thermal radiation contour
- [ ] Result table with thresholds

### Phase 7: Explosion (2 weeks)
- [ ] TNT equivalency (Kingery-Bulmash)
- [ ] TNO Multi-Energy (blast curve interpolation)
- [ ] Baker-Strehlow-Tang
- [ ] Confinement auto-assessment tool
- [ ] Obstacle effects (reflection, diffraction)
- [ ] Explosion input panel (UI)
- [ ] Overpressure contour
- [ ] Result table

### Phase 8: Terrain & 3D (2 weeks)
- [ ] DEM loader (GeoTIFF, HGT, ASCII Grid)
- [ ] Obstacle/building model (bounding box list)
- [ ] Line-of-sight engine (ray casting)
- [ ] 3D visualization (PyVista)
- [ ] Terrain input dialog (UI)
- [ ] Obstacle editor (draw/list)
- [ ] 3D contour viewer

### Phase 9: Toxicology & Vulnerability (1-2 weeks)
- [ ] Probit models (thermal, overpressure, toxic)
- [ ] Toxic load Cⁿ·t (dose-based)
- [ ] Shelter factor (indoor protection)
- [ ] Evacuation time window
- [ ] Multi-species (human, environment)
- [ ] Vulnerability result tab

### Phase 10: QRA Framework (2-3 weeks)
- [ ] Failure frequency database
- [ ] Event tree analysis (ETA)
- [ ] Ignition probability model
- [ ] Explosion probability model
- [ ] Individual risk (IRPA) calculation
- [ ] Societal risk (FN curve)
- [ ] Risk matrix (ISO 17776)
- [ ] ALARP demonstration
- [ ] Population data input (day/night, indoor/outdoor)
- [ ] QRA input panel (UI)
- [ ] IR contour visualization
- [ ] FN curve chart
- [ ] Risk matrix chart

### Phase 11: Advanced Analysis (2 weeks)
- [ ] Batch runner (multi-case, multi-weather)
- [ ] Sensitivity analysis (tornado chart)
- [ ] Monte Carlo uncertainty (distribution sampling)
- [ ] Worst-case identification
- [ ] Batch run dialog (UI)
- [ ] Sensitivity config dialog
- [ ] Monte Carlo config dialog
- [ ] Parallel processing (multiprocessing pool)

### Phase 12: Multi-Case, Reporting & Export (2 weeks)
- [ ] Multi-case overlay engine + UI
- [ ] Case comparison table
- [ ] PDF report generator (full)
- [ ] Excel export (multi-sheet, openpyxl)
- [ ] Text/CSV export
- [ ] GIS export (GeoJSON, Shapefile, KML)
- [ ] PNG/SVG image export
- [ ] Map tile basemap (Cartopy)

### Phase 13: Audit Trail & File Management (1 week)
- [ ] Audit trail engine (version log, diffs)
- [ ] Audit viewer UI
- [ ] Rollback capability
- [ ] Checkpoint system

### Phase 14: Validation & Benchmarking (2 weeks)
- [ ] Validation against published benchmark cases
  - FLADIS (dense gas, ammonia)
  - Desert Tortoise (ammonia)
  - Maplin Sands (LNG)
  - Coyote (LNG)
  - Thorney Island (heavy gas)
  - Published pool fire data
  - Published explosion test data
- [ ] Run all 36 test cases from TEST_CASES.md
- [ ] Regression test suite automation
- [ ] Fix all validation failures
- [ ] Performance benchmarking

### Phase 15: Documentation (2 weeks)
- [ ] Installation Manual (INSTALLATION.md) — full, with screenshots
- [ ] User Manual (USER_MANUAL.md) — all 14 parts complete
- [ ] Tutorial (TUTORIAL.md) — 8 tutorial projects, step-by-step
- [ ] Test Cases (TEST_CASES.md) — 36 cases with inputs/outputs/tolerances
- [ ] Methodology Reference (METHODOLOGY.md) — all equations & model descriptions
- [ ] Help system integration: F1 mapping, tooltips, "?" buttons
- [ ] In-app searchable help viewer
- [ ] Screenshots & diagrams for docs

### Phase 16: Test Case Implementation & CI (2 weeks)
- [ ] Implement 36 automated test cases in pytest
  - TC-D01 to TC-D10: Dispersion tests
  - TC-F01 to TC-F08: Fire tests
  - TC-E01 to TC-E07: Explosion tests
  - TC-S01 to TC-S05: Source term tests
  - TC-Q01 to TC-Q03: QRA tests
  - TC-M01 to TC-M03: Misc tests
- [ ] CI pipeline: all tests must pass before merge
- [ ] Coverage report (target: >80%)
- [ ] Test data fixtures

### Phase 17: Package & Deploy (2 weeks)
- [ ] Auto-updater (GitHub Releases API)
- [ ] PyInstaller packaging
- [ ] Windows installer (.msi / .exe) + code signing
- [ ] Linux AppImage
- [ ] macOS .dmg (optional)
- [ ] GitHub Releases + changelog
- [ ] Version tagging (semver: 1.0.0)

**Total estimasi: ~30-38 weeks (7-9 months)**

---

## 9. System Requirements

| | Development | Runtime (Minimum) | Runtime (Recommended) |
|---|---|---|---|
| OS | Any (Python) | Windows 10/11, Linux, macOS 12+ | Windows 10/11 x64 |
| Python | 3.11+ | (bundled) | — |
| RAM | 16 GB | 4 GB | 8-16 GB |
| CPU | Multi-core | 2 cores | 4+ cores (batch/Monte Carlo) |
| Disk | 1 GB | 500 MB | 2 GB (with DEM data) |
| GPU | Optional | Not required | For 3D rendering |
| Display | 1920×1080 | 1366×768 | 1920×1080+ |

---

## 10. Decisions (Confirmed)

| Item | Decision |
|---|---|
| **Software Name** | **Rekarisk** |
| Logo/Branding | Placeholder (kosong) — user bisa set nanti |
| UI Language | **English** |
| Project File Format | **JSON (*.caproj)** — human-readable, versionable |
| Multi-Case Overlay | **Yes** — overlay N cases, per-case legend |
| Custom Substance | **Yes** — pure substances + **mixtures (composition)** |
| Units | **Metric (SI) default**, imperial via settings |
| Auto-Updater | **Yes** — GitHub Releases + SHA256 verify |
| File Operations | **New / Open / Save / Save As** |
| Export Formats | Excel, Text/CSV, PDF, GeoJSON, Shapefile, KML, PNG, SVG |
| Source Term Engine | **Included** — this is mandatory for real-world use |
| QRA Framework | **Included** — what Phast+SAFETI does |
| Advanced Met | **Included** — wind rose, joint probability |
| 3D + Terrain | **Included** — DEM + obstacles + LOS |
| Equation of State | **Included** — Peng-Robinson + VLE flash |
| Toxic Dose | **Included** — Cⁿ·t + shelter factor |
| Batch / Sensitivity / Monte Carlo | **Included** |
| Audit Trail | **Included** |
| Documentation | Installation Manual, User Manual, Tutorial, 36 Test Cases, Help System |
| GitHub Repository | `github.com/arienug2000/rekarisk` |
| Execution Start | **Tonight, 11 PM WIB / 23:00 UTC+7** |

---

## 11. GitHub & Repository Setup

```
Repository: github.com/arienug2000/rekarisk

Branch structure:
  main          — stable, protected, only via PR
  develop       — integration branch
  feature/*     — per-phase feature branches (feature/source-term, feature/dispersion, etc.)
  docs/*        — documentation branches
  fix/*         — bug fixes

Commit convention: Conventional Commits
  feat: add Gaussian plume model
  fix: correct pool fire view factor
  docs: add installation guide
  test: add TC-D03 puff model test
  refactor: restructure substance database

CI/CD (GitHub Actions):
  - On push/PR: pytest, lint (flake8/ruff), type check (mypy)
  - On tag v*: build PyInstaller + create GitHub Release

Release flow:
  1. Develop in feature/*
  2. PR → develop (CI must pass)
  3. develop → main (PR review)
  4. Tag main: v1.0.0-beta.1, v1.0.0-rc.1, v1.0.0
  5. GitHub Action builds & uploads release artifacts
```

---

## 12. Phast / SAFETI Feature Comparison

| Module | Phast | SAFETI | This Software |
|---|---|---|---|
| **Source term** | ✅✅ Full discharge engine | (from Phast link) | ✅ Orifice, pipe, vessel, two-phase, pool, rainout |
| **Dispersion** | ✅ UDM (Unified Dispersion Model) | (from Phast) | ✅ Gaussian + dense gas + building wake |
| **Pool fire** | ✅ | — | ✅ Multi-correlation |
| **Jet fire** | ✅ API 521 | — | ✅ API 521 |
| **BLEVE** | ✅ | — | ✅ HSE + CCPS |
| **Flash fire** | ✅ | — | ✅ LFL isopleth |
| **VCE** | ✅ TNO + BST | — | ✅ TNT + TNO + BST |
| **Toxicology** | ✅ Probit + dose | — | ✅ Probit + Cⁿ·t load |
| **3D terrain** | ✅ DEM + obstacles | — | ✅ DEM + obstacles + LOS |
| **Wind rose** | ✅ Joint prob | ✅ | ✅ Wind rose + JPD |
| **QRA — event tree** | ✅ | ✅✅ | ✅ ETA + ignition/explosion prob |
| **QRA — IRPA** | — | ✅✅ | ✅ IRPA contour |
| **QRA — FN curve** | — | ✅✅ | ✅ FN + criteria |
| **QRA — risk matrix** | — | ✅✅ | ✅ ISO 17776 matrix |
| **ALARP** | — | ✅✅ | ✅ Cost-benefit |
| **Population** | — | ✅✅ Day/night | ✅ Day/night + indoor fraction |
| **Batch run** | ✅ | ✅ | ✅ Multiprocessing |
| **Sensitivity** | — | ✅ | ✅ Tornado chart |
| **Monte Carlo** | — | ✅ (limited) | ✅ Full MC |
| **Audit trail** | ✅ (limited) | ✅ | ✅ Full versioning |
| **GIS export** | ✅ Shapefile | ✅ Shapefile | ✅ GeoJSON + Shapefile + KML |
| **Reports** | ✅ PDF/Word | ✅ | ✅ PDF + Excel + CSV |
| **Custom substance** | ✅ | ✅ | ✅ Pure + mixtures |
| **Equation of state** | ✅ PR/SRK | ✅ | ✅ PR + SRK |
| **DIPPR props** | ✅ (DIPPR DB) | ✅ | ✅ DIPPR correlations |
| **Multi-case overlay** | ✅ | ✅ | ✅ |
| **License cost** | $$$$/yr | $$$$/yr | **Open source** ✅ |

---

## 13. Documentation & Help Suite

### 12A. Installation Manual

```
File: docs/INSTALLATION.md

Contents:
  1. System Requirements
     - OS: Windows 10+, Linux (Ubuntu 22.04+, Debian 12+), macOS 12+
     - RAM: 4 GB minimum, 8 GB recommended
     - Disk: 500 MB (software) + 2 GB (DEM data, optional)
  2. Quick Install (Windows)
     - Download .exe installer from GitHub Releases
     - Run installer → Next → Next → Finish
     - Launch from Start Menu or Desktop shortcut
  3. Quick Install (Linux)
     - Download .AppImage
     - chmod +x Rekarisk-*.AppImage
     - ./Rekarisk-*.AppImage
  4. Quick Install (macOS)
     - Download .dmg
     - Drag to Applications
     - Right-click → Open (first time, Gatekeeper)
  5. Install from Source (Developers)
     - git clone + requirements.txt
     - Python 3.11+ setup
     - pip install -e .
  6. First Launch
     - License agreement
     - Default project location
     - Check for updates
  7. Uninstall
     - Windows: Add/Remove Programs
     - Linux: delete AppImage
     - macOS: drag to Trash
  8. Troubleshooting
     - Missing DLL (Windows)
     - Display scaling issues
     - Permission errors
```

### 12B. User Manual

```
File: docs/USER_MANUAL.md (English, comprehensive)

Structure (per main function):

  Part 0: Introduction
    - What is Rekarisk?
    - Consequence Analysis basics
    - QRA basics
    - Regulatory framework (API, Kepmen, CCPS)
    - Software overview (main window tour)

  Part 1: Getting Started
    - Creating a new project
    - Saving and opening projects
    - Project navigator
    - Units configuration

  Part 2: Substance Database
    - Browsing built-in substances
    - Adding custom pure substances
    - Creating mixtures with composition
    - Properties overview

  Part 3: Source Term
    - Release scenarios overview
    - Orifice leak: liquid, gas, vapor
    - Two-phase release
    - Pipe rupture
    - Vessel blowdown
    - Relief valve discharge
    - Pool spreading & evaporation
    - Rainout & aerosol
    - Interpreting source term results

  Part 4: Meteorology
    - Pasquill stability classes
    - Single weather case
    - Wind rose & probability table
    - Diurnal variation
    - Seasonal profiles
    - Importing weather data

  Part 5: Dispersion Modeling
    - Gaussian plume (continuous)
    - Gaussian puff (instantaneous)
    - Dense gas dispersion
    - Building wake effects
    - Indoor/outdoor infiltration
    - Concentration contours & interpretation

  Part 6: Fire Modeling
    - Pool fire
    - Jet fire
    - BLEVE / Fireball
    - Flash fire
    - Line-of-sight blocking
    - Thermal radiation contours

  Part 7: Explosion Modeling
    - TNT equivalency
    - TNO Multi-Energy
    - Baker-Strehlow-Tang
    - Confinement assessment
    - Obstacle effects
    - Overpressure contours

  Part 8: Terrain & 3D
    - Importing DEM
    - Adding buildings/obstacles
    - 3D visualization
    - Line-of-sight setup

  Part 9: Toxicology & Vulnerability
    - Probit models
    - Toxic load (dose-based)
    - Shelter factor
    - Evacuation time window

  Part 10: QRA (Quantitative Risk Assessment)
    - Failure frequency database
    - Event tree analysis
    - Individual risk (IRPA)
    - Societal risk (FN curve)
    - Risk matrix
    - ALARP demonstration
    - Population data

  Part 11: Advanced Analysis
    - Batch running
    - Sensitivity analysis (tornado charts)
    - Monte Carlo uncertainty
    - Worst-case identification

  Part 12: Output & Reporting
    - Contour maps (2D/3D)
    - Result tables
    - PDF report generation
    - Excel export
    - GIS export (GeoJSON, Shapefile, KML)

  Part 13: Multi-Case Comparison
    - Overlay multiple cases
    - Side-by-side comparison
    - Worst-case from multiple scenarios

  Part 14: Audit Trail
    - Viewing change history
    - Comparing versions
    - Rollback

  Appendices:
    A. Glossary of terms
    B. All equations & methodology reference
    C. Regulatory threshold tables
    D. Substance database listing
    E. Keyboard shortcuts
    F. Error messages & troubleshooting
```

### 12C. Tutorial (Built-in, Interactive)

```
File: docs/TUTORIAL.md (step-by-step, project-based)

8 Tutorial Projects, each ~20-30 min:

  Tutorial 1: First Steps
    - Open Rekarisk
    - Create new project
    - Browse substance database
    - Save project

  Tutorial 2: Simple Dispersion
    - Chlorine leak from storage tank
    - Gaussian plume model
    - Interpret concentration contours
    - Export results

  Tutorial 3: Source Term + Dispersion
    - Calculate release rate first
    - Feed to dispersion
    - Compare: manual rate vs calculated rate

  Tutorial 4: Pool Fire
    - LNG spill on water
    - Radiant heat calculation
    - Hazard zone determination
    - Multi-threshold contour

  Tutorial 5: BLEVE & Jet Fire
    - LPG tank BLEVE
    - Fireball radiation
    - Jet fire from pipeline rupture

  Tutorial 6: Explosion
    - VCE in process area
    - TNO Multi-Energy (choose strength)
    - Overpressure contour
    - Building damage assessment

  Tutorial 7: QRA Basics
    - Release scenario + frequency
    - Event tree
    - Risk contour
    - Interpret FN curve

  Tutorial 8: Multi-Case & Batch
    - 3 hole sizes + 4 weather conditions
    - Batch run (12 cases)
    - Worst-case identification
    - Export comparison table
```

### 12D. Test Cases (30+ Validated Scenarios)

Each test case includes:
- Unique ID
- Model tested
- Input parameters
- Expected output (benchmark value)
- Source/reference
- Tolerance (pass/fail criteria)

```
File: docs/TEST_CASES.md

═══════════════════════════════════════════════════
DISPERSION TEST CASES (10 cases)
═══════════════════════════════════════════════════

TC-D01: Gaussian Plume — Basic Validation
  Model: Gaussian plume, neutral buoyancy
  Substance: SO₂
  Release: 5 kg/s continuous, 20m height
  Weather: D-stability, 5 m/s wind
  Benchmark: C_max (ground) ≈ 3,980 μg/m³ at x ≈ 800m
  Reference: Turner Workbook, Problem 3.1
  Tolerance: ±10%

TC-D02: Gaussian Plume — Ground Release
  Model: Gaussian plume, total reflection
  Substance: CO (non-buoyant)
  Release: 10 kg/s, H=0m
  Weather: D-stability, 3 m/s
  Benchmark: C(x=500m, y=0, z=0) ≈ 70,700 μg/m³
  Reference: Turner Workbook
  Tolerance: ±10%

TC-D03: Gaussian Puff — Instantaneous Release
  Model: Gaussian puff, single puff
  Substance: SO₂
  Mass: 100 kg instantaneous
  Weather: D-stability, 4 m/s
  Benchmark: C_max at x=300m ≈ 2,650 μg/m³, arrival ~75s
  Reference: CCPS Guidelines, Example 2.5
  Tolerance: ±15%

TC-D04: Dense Gas — Chlorine Release
  Model: Dense gas (SLAB-based)
  Substance: Cl₂
  Release: 10 kg/s, ground level
  Weather: F-stability, 2 m/s
  Benchmark: LFL-like (20 ppm) distance ≈ 2,100m
  Reference: HSE Chlorine Release Model
  Tolerance: ±20%

TC-D05: Dense Gas — Ammonia
  Model: Dense gas
  Substance: NH₃
  Release: 50 kg/s, instantaneous
  Weather: D-stability, 5 m/s
  Benchmark: IDLH (300 ppm) distance ≈ 1,500m
  Reference: FLADIS field trial (scaled)
  Tolerance: ±20%

TC-D06: Dense Gas — LNG
  Model: Dense gas + lift-off
  Substance: Methane (LNG)
  Release: 100 kg/s, cryogenic
  Weather: D-stability, 3 m/s
  Benchmark: ½ LFL distance ≈ 600m; lift-off ~400m
  Reference: Maplin Sands trial
  Tolerance: ±25%

TC-D07: Building Wake — Cavity
  Model: Building downwash
  Substance: non-reactive tracer
  Building: 20m × 10m × 10m
  Stack: H=15m (below 1.5× building H)
  Benchmark: Cavity C/C₀ ≈ 0.5-1.0 at x=2H
  Reference: ASHRAE Handbook
  Tolerance: ±30%

TC-D08: Buoyant Plume Rise
  Model: Briggs plume rise
  Substance: Hot combustion products
  Stack: exit T=450K, ambient=293K, exit V=10 m/s
  Benchmark: ΔH ≈ 45m (plume rise)
  Reference: Briggs (1969)
  Tolerance: ±20%

TC-D09: Indoor Concentration
  Model: Infiltration, ACH-based
  Outdoor C: 100 ppm constant
  ACH: 2 hr⁻¹
  Benchmark: Indoor/Outdoor ratio → 1.0 after ~0.5 hr
  Reference: Mass balance equation
  Tolerance: ±5%

TC-D10: Crosswind Profile
  Model: Gaussian plume cross-section
  Substance: Tracer
  Release: 1 kg/s, H=10m
  Crosswind at x=500m, D-stability, 5 m/s
  Benchmark: σ_y at x=500m ≈ 36m (rural), C at y=σ_y/C_center ≈ 0.606
  Reference: Pasquill-Gifford curves
  Tolerance: ±10%

═══════════════════════════════════════════════════
FIRE TEST CASES (8 cases)
═══════════════════════════════════════════════════

TC-F01: Pool Fire — LNG (Small)
  Model: Pool fire, Shokri-Beyler
  Substance: LNG (methane)
  Pool diameter: 5m
  Wind: 3 m/s, ambient 298K, RH 70%
  Benchmark: Heat flux at 50m ≈ 1.6 kW/m²
  Reference: Mudan (1984), CCPS example
  Tolerance: ±20%

TC-F02: Pool Fire — Gasoline (Large)
  Model: Pool fire, large diameter
  Substance: Gasoline (n-octane)
  Pool diameter: 20m
  Wind: 0 m/s (calm)
  Benchmark: Flame height ≈ 35m; SEP ≈ 120 kW/m²
  Reference: LNGFIRE3, HSE
  Tolerance: ±20%

TC-F03: Pool Fire — Smoke Obscuration
  Model: Smoke factor
  Substance: Crude oil (sooty)
  Pool diameter: 10m
  Benchmark: SEP reduced by ~50% due to smoke
  Reference: TNO Yellow Book
  Tolerance: ±30%

TC-F04: Jet Fire — Sonic (Methane)
  Model: API RP 521
  Substance: Methane
  Orifice: 25mm, P=50 bar, sonic flow
  Benchmark: Flame length ≈ 27m; heat flux at 50m ≈ 4 kW/m²
  Reference: Chamberlain (1987)
  Tolerance: ±20%

TC-F05: Jet Fire — Subsonic (Propane)
  Model: API RP 521, subsonic
  Substance: Propane
  Orifice: 10mm, P=5 bar, subsonic
  Benchmark: Flame length ≈ 5m
  Reference: API 521 Annex
  Tolerance: ±20%

TC-F06: BLEVE — LPG 50 ton
  Model: Fireball (HSE)
  Substance: Propane (LPG)
  Mass: 50,000 kg
  Benchmark: Fireball diameter ≈ 213m; duration ≈ 16s
  Reference: HSE, CCPS
  Tolerance: ±15%

TC-F07: BLEVE — Small Cylinder
  Model: Fireball (CCPS)
  Substance: Propane
  Mass: 500 kg
  Benchmark: Fireball diameter ≈ 47m; duration ≈ 4.7s
  Reference: CCPS Guidelines
  Tolerance: ±15%

TC-F08: Flash Fire — LFL Envelope
  Model: Dispersion + LFL contour
  Substance: Propane
  Release: 10 kg/s, P=10 bar
  Weather: F-stability, 2 m/s
  Benchmark: LFL (2.1%) distance ≈ 250m
  Reference: Phast comparison
  Tolerance: ±25%

═══════════════════════════════════════════════════
EXPLOSION TEST CASES (7 cases)
═══════════════════════════════════════════════════

TC-E01: TNT — 1000 kg
  Model: TNT equivalency
  TNT mass: 1000 kg
  Benchmark: P_s at 50m ≈ 0.64 bar; P_s at 100m ≈ 0.18 bar
  Reference: Kingery-Bulmash (1984)
  Tolerance: ±10%

TC-E02: TNT — Propane VCE (small η)
  Model: TNT equivalency
  Substance: Propane, mass=1000 kg, η=0.03
  M_TNT: 1000 × 46350 / 4680 × 0.03 ≈ 297 kg
  Benchmark: P_s at 50m ≈ 0.19 bar
  Reference: CCPS example
  Tolerance: ±15%

TC-E03: TNO — Strength 7 (Highly Confined)
  Model: TNO Multi-Energy
  Combustible mass: 5000 kg propane
  Strength: 7, congested region R=50m
  Benchmark: P_s at 100m ≈ 0.3 bar
  Reference: TNO Yellow Book
  Tolerance: ±20%

TC-E04: TNO — Strength 4 (Partially Confined)
  Model: TNO Multi-Energy
  Combustible mass: 5000 kg
  Strength: 4, congested region R=75m
  Benchmark: P_s at 100m ≈ 0.05 bar
  Reference: TNO Yellow Book
  Tolerance: ±25%

TC-E05: TNO — Strength 10 (Detonation)
  Model: TNO Multi-Energy
  Combustible mass: 1000 kg
  Strength: 10
  Benchmark: P_s at 50m ≈ 0.82 bar
  Reference: TNO Yellow Book
  Tolerance: ±15%

TC-E06: BST — High Congestion, Medium Reactivity
  Model: Baker-Strehlow
  Fuel: Propane (medium reactivity)
  Congestion: High, Confinement: 2D
  Expected M_f ≈ 0.5
  Benchmark: P_s at scaled distance R̄=1 ≈ 0.3 bar
  Reference: Baker et al. (1997)
  Tolerance: ±20%

TC-E07: BST — Low Congestion, Low Reactivity
  Model: Baker-Strehlow
  Fuel: Methane (low reactivity)
  Congestion: Low, Confinement: 1D
  Expected M_f ≈ 0.2
  Benchmark: P_s at scaled distance R̄=1 ≈ 0.03 bar
  Reference: Baker et al.
  Tolerance: +30% / -20%

═══════════════════════════════════════════════════
SOURCE TERM TEST CASES (5 cases)
═══════════════════════════════════════════════════

TC-S01: Liquid Leak — Water
  Model: Bernoulli orifice
  Substance: Water
  Vessel: P=5 barg, H_liquid=10m, hole=10mm
  Benchmark: Q ≈ 0.63 kg/s (C_d=0.62)
  Reference: Bernoulli equation
  Tolerance: ±5%

TC-S02: Gas Leak — Sonic (Methane)
  Model: Choked gas orifice
  Substance: Methane
  Vessel: P=50 barg, T=298K, hole=25mm
  Benchmark: Q ≈ 9.8 kg/s
  Reference: API 520 / Crane
  Tolerance: ±10%

TC-S03: Two-Phase — Propane
  Model: HEM two-phase
  Substance: Propane (saturated liquid, T=298K)
  Vessel: P_sat≈9.5 bar, hole=25mm
  Benchmark: Q ≈ 13.5 kg/s; vapor mass fraction ≈ 0.35
  Reference: Leung / API 520
  Tolerance: ±20%

TC-S04: Pool Evaporation
  Model: Mackay-Matsugu
  Substance: LNG (methane), cryogenic
  Pool area: 100 m²
  Benchmark: Evaporation rate ≈ 0.05 kg/m²·s (cryogenic boiling)
  Reference: CCPS / TNO
  Tolerance: ±25%

TC-S05: Vessel Blowdown — Gas
  Model: Vessel depressurization
  Substance: Nitrogen
  Vessel: V=10 m³, P_initial=50 bar, hole=10mm
  Benchmark: P(60s) ≈ 43 bar, total mass lost ≈ 35 kg
  Reference: Energy balance model
  Tolerance: ±15%

═══════════════════════════════════════════════════
QRA TEST CASES (3 cases)
═══════════════════════════════════════════════════

TC-Q01: Event Tree — Single Scenario
  Model: ETA
  Release f=1×10⁻⁴/yr, p_imm_ign=0.1, p_del_ign=0.2, p_exp=0.5
  Benchmark: P(jet fire)=1×10⁻⁵/yr, P(VCE)=9×10⁻⁶/yr, P(safe)=7.2×10⁻⁵/yr
  Reference: Probability multiplication
  Tolerance: ±1%

TC-Q02: FN Curve — 3 Scenarios
  Model: Societal risk
  Scenario A: f=1e-4/yr, N=10
  Scenario B: f=1e-5/yr, N=100
  Scenario C: f=1e-6/yr, N=5
  Benchmark: F(N≥5)=1.11e-4, F(N≥10)=1.1e-5, F(N≥100)=1e-5
  Reference: CCPS CPQRA
  Tolerance: ±1%

TC-Q03: Risk Matrix
  Model: ISO 17776 matrix
  Consequence=Cat-4 (major), Likelihood=C (1e-4 to 1e-3)
  Benchmark: Matrix cell = "ALARP" (yellow)
  Reference: ISO 17776
  Tolerance: Exact match

═══════════════════════════════════════════════════
MISCELLANEOUS TEST CASES (3 cases)
═══════════════════════════════════════════════════

TC-M01: Mixture — LNG (multicomponent)
  Model: Mixture EoS + dispersion
  Mixture: 85% CH₄, 10% C₂H₆, 5% C₃H₈
  Benchmark: MW ≈ 18.7, NBP ≈ 114K (weighted)
  Reference: Le Chatelier, mixing rules
  Tolerance: ±5%

TC-M02: Units Conversion — All
  Model: Units module
  Test: 1 bar → 100,000 Pa → 14.504 psi → 0.9869 atm → 750.06 mmHg
  Benchmark: All within 0.1%
  Reference: NIST
  Tolerance: ±0.1%

TC-M03: Probit — Thermal Dose
  Model: Thermal probit
  Dose: I^(4/3)·t = 1000 (kW/m²)^(4/3)·s
  Benchmark: Pr = -14.9 + 2.56·ln(1000) ≈ 2.78
  P_fatality = Φ(2.78-5) = Φ(-2.22) ≈ 0.013
  Reference: TNO Green Book
  Tolerance: ±5%

═══════════════════════════════════════════════════
SUMMARY: 36 Test Cases
═══════════════════════════════════════════════════

| Module | Count | IDs |
|--------|-------|-----|
| Dispersion | 10 | TC-D01 to TC-D10 |
| Fire | 8 | TC-F01 to TC-F08 |
| Explosion | 7 | TC-E01 to TC-E07 |
| Source Term | 5 | TC-S01 to TC-S05 |
| QRA | 3 | TC-Q01 to TC-Q03 |
| Misc | 3 | TC-M01 to TC-M03 |
| **TOTAL** | **36** | |
```

### 12E. Context-Sensitive Help (F1)

```
Built-in Help System:
  - F1 on any panel → jump to relevant manual section
  - Tooltips on every input field with:
    - Parameter description
    - Valid range
    - Typical values
    - Unit
  - "?" button on each panel → mini explainer dialog
  - Status bar: brief help text on hover
  - Help → Search: full-text search across manual
  - Error messages include link to relevant doc section

Help data format:
  help/
    help_topics.json     # Topic index
    help_content/        # HTML/Markdown per topic
    help_images/         # Diagrams & screenshots
```

---

## 14. Open Questions (Final)

1. **Mixture phase envelope** — perlu sampai sejauh mana? Full 3-phase (V-L-H) atau cukup V-L?
2. **QRA failure frequency** — gunakan data generik (HSE/OGP published) atau biarkan user input sendiri semua frekuensi?
3. **DEM auto-download** — fitur download otomatis SRTM tiles dari internet (butuh koneksi) atau manual import aja?
4. **3D Plume** — real-time interactive (butuh GPU) atau static render cukup?
5. **Parallel processing** — gunakan multiprocessing lokal atau support juga distribute ke network?
6. **Water-reactive chemicals** — seberapa detail? Cukup HCl dari TiCl₄ atau perlu database reaksi lengkap?
7. **Populasi grid** — user input manual per cell atau impor dari file?
8. **Open-source license** — GPLv3, MIT, atau Apache 2.0?
9. **Logo** — placeholder dulu atau mau bikin logo "Rekarisk"?

---

*Plan v3 (Final) — Rekarisk. Semua keputusan sudah masuk, 36 test cases terdefinisi, documentation suite lengkap. Fase 0 dimulai malam ini pukul 23:00 WIB.*
