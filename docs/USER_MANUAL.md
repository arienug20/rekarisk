# Rekarisk User Manual

## 1. Getting Started

### 1.1 Installation

```bash
# Clone the repository
git clone https://github.com/arienug20/rekarisk.git
cd rekarisk

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Rekarisk in development mode
pip install -e .
```

**Dependencies:**
- Python 3.9+
- PyQt6 (GUI framework)
- NumPy, SciPy (numerical computation)
- Matplotlib (plotting)
- CoolProp (optional, for EoS validation)

### 1.2 First Run

```bash
python -m rekarisk
```

Or launch from the installed entry point:

```bash
rekarisk
```

The main window presents a dashboard with quick-launch cards for common
analysis types and a menu bar for navigation.

---

## 2. Creating a Project

### 2.1 New Project

1. Click **File → New Project** or press `Ctrl+N`
2. Enter project name and description
3. Select working directory
4. Click **Create**

### 2.2 Project Structure

Each project contains:
- **Scenarios** — individual hazard scenarios (release, fire, explosion)
- **Substances** — chemicals referenced by scenarios
- **Meteorology** — weather data sets
- **Maps** — optional geographical context (GIS layer)
- **QRA** — risk assessment integration

### 2.3 Opening Existing Projects

Click **File → Open Project** or select from recent projects list.

---

## 3. Substance Database

### 3.1 Browsing Substances

Navigate to **Database → Substances** to view the built-in library of
common chemicals including:
- Methane, Ethane, Propane, Butane
- Hydrogen
- Ammonia, Chlorine
- Gasoline, Diesel, Kerosene
- Methanol, Ethanol

### 3.2 Searching

Use the search bar to filter by name, CAS number, or formula.

### 3.3 Adding Custom Substances

1. Click **Add Substance**
2. Fill in required properties:
   - Name, CAS number, molecular formula
   - Molecular weight (kg/mol)
   - Boiling point (K)
   - Heat of vaporization (J/kg)
   - Heat of combustion (J/kg)
   - Vapor pressure coefficients
   - Probit constants (a, b, n)

### 3.4 Editing Substances

Select a substance and click **Edit**. Changes are saved to user database
(preserved across projects).

---

## 4. Source Term Analysis

### 4.1 Orifice Discharge

1. Navigate to **Source Term → Orifice Discharge**
2. Select or enter:
   - Substance from database
   - Orifice diameter and discharge coefficient
   - Upstream pressure and temperature
   - Downstream pressure
   - Phase (liquid/gas/two-phase)
3. Click **Calculate**

Results panel shows:
- Mass flow rate (kg/s)
- Volumetric flow rate
- Velocity (m/s)
- Flow regime (subsonic/choked)
- Gas expansion factor (for gas discharge)

### 4.2 Vessel Blowdown

1. Navigate to **Source Term → Vessel Blowdown**
2. Enter:
   - Vessel volume and wall area
   - Initial pressure and temperature
   - Orifice specifications
   - Simulation time and pressure target
   - Phase (gas/two-phase)
3. Click **Simulate**

Results:
- Time history plot: P(t), T(t), mdot(t)
- Total mass released
- Maximum release rate
- Blowdown duration

### 4.3 PSV Sizing

1. Navigate to **Source Term → Relief Valve Sizing**
2. Select scenario (fire exposure, blocked outlet, thermal expansion, etc.)
3. Enter set pressure, relieving temperature, flow rate
4. Select fluid type (gas/vapor/liquid/steam)
5. Click **Size**

The tool calculates:
- Required orifice area (mm²)
- Selected API orifice designation (D through T)
- Is the flow choked?
- Backpressure correction

### 4.4 Pool Evaporation

1. Navigate to **Source Term → Pool Evaporation**
2. Enter spill mass, pool surface type (concrete/soil/water)
3. Set bund area if applicable
4. Enter weather conditions (wind speed, ambient temperature)
5. Click **Simulate**

Output:
- Pool radius/area vs time
- Evaporation rate vs time
- Total mass evaporated
- Time to complete evaporation

---

## 5. Dispersion Analysis

### 5.1 Gaussian Plume

1. Navigate to **Dispersion → Gaussian Plume**
2. Set source:
   - Source rate (kg/s)
   - Release height
3. Set meteorology:
   - Wind speed at 10m
   - Stability class (A–F)
   - Terrain type (rural/urban)
4. Define computational grid (or use defaults)
5. Click **Run**

Results:
- Color contour plot of concentration field
- Centerline concentration vs distance
- Table: concentration at receptor points
- Isopleth areas (flammable/toxic)

### 5.2 Gaussian Puff

1. Navigate to **Dispersion → Gaussian Puff**
2. Set release mass (instantaneous)
3. Configure wind speed, direction, stability
4. Set time range for animation/grid
5. Click **Run**

The puff visualization shows:
- Puff concentration snapshots at selected times
- Peak concentration track
- Time of arrival at receptors

### 5.3 Dense Gas

1. Navigate to **Dispersion → Dense Gas**
2. Select release type (instantaneous/continuous)
3. Enter initial cloud parameters (mass, radius, height)
4. Set density ratio (>1 = heavier than air)
5. Click **Simulate**

Results show cloud geometry evolution:
- Radius vs time
- Height vs time
- Concentration vs time
- Transition to passive dispersion point

---

## 6. Fire Analysis

### 6.1 Pool Fire

1. Navigate to **Fire → Pool Fire**
2. Select substance (or enter burn rate params)
3. Enter pool diameter or area (use pool evaporation output)
4. Set wind speed
5. Click **Calculate**

Results:
- Burning rate (kg/m²·s)
- Flame geometry (length, tilt angle)
- Surface emissive power (SEP)
- Thermal radiation vs distance plot
- Distance to threshold (37.5, 12.5, 5.0, 1.6 kW/m²)

### 6.2 Jet Fire

1. Navigate to **Fire → Jet Fire**
2. Enter release parameters:
   - Orifice diameter
   - Mass flow rate (use orifice discharge output)
   - Release pressure and temperature
3. Select gas (for heat of combustion)
4. Click **Calculate**

### 6.3 BLEVE / Fireball

1. Navigate to **Fire → BLEVE**
2. Enter mass in vessel and substance
3. Optionally set failure fraction (< 1 for partial BLEVE)
4. Click **Calculate**

Results:
- Fireball diameter and duration
- Maximum surface emissive power
- Thermal radiation vs distance
- Threshold distances

---

## 7. Explosion Analysis

### 7.1 TNT Equivalency

1. Navigate to **Explosion → TNT Equivalency**
2. Enter combustible mass
3. Set efficiency factor (typically 0.01–0.10)
4. Optionally provide heat of combustion (or select from substance DB)
5. Click **Calculate**

Output:
- TNT equivalent mass (kg)
- Overpressure vs distance table and plot
- Distance to overpressure thresholds (1, 3.5, 7, 21, 35, 70 kPa)
- Damage level contours

### 7.2 TNO Multi-Energy

1. Navigate to **Explosion → TNO Multi-Energy**
2. Enter combustible mass and heat of combustion
3. Select blast strength (1–10) or auto-estimate from:
   - Congestion level (low/medium/high)
   - Confinement (unconfined/partly confined/confined)
4. Click **Calculate**

### 7.3 Baker-Strehlow-Tang

1. Navigate to **Explosion → BST**
2. Enter mass and fuel
3. Select or auto-assign:
   - Reactivity category (low/medium/high)
   - Mach number
4. Click **Calculate**

---

## 8. QRA (Quantitative Risk Assessment)

### 8.1 Event Trees

1. Navigate to **QRA → Event Trees**
2. Select template or build custom:
   - **Vessel Release** (immediate/delayed ignition → fireball/flash fire/dispersion)
   - **Pipeline Rupture** (ignition → jet fire/dispersion)
   - **Tank Spill** (pool formation → ignition → pool fire/dispersion)
3. Set initiating frequency
4. Set branch probabilities
5. Click **Calculate** to enumerate scenarios

Each scenario shows:
- Description
- Frequency (/year)
- Consequence type (fireball, flash fire, dispersion, explosion)

### 8.2 Individual Risk

1. Navigate to **QRA → Individual Risk**
2. Combine event tree scenarios with consequence models
3. Define grid or receptor points
4. Click **Calculate IRPA**

Output:
- IRPA contours (1×10⁻⁴, 1×10⁻⁵, 1×10⁻⁶, 1×10⁻⁷ /year)
- Table: IRPA at each receptor

### 8.3 Societal Risk (FN Curve)

1. Navigate to **QRA → Societal Risk**
2. Define population distribution on map grid
3. Click **Calculate FN Curve**

The FN plot shows:
- Cumulative frequency F(N) vs number of fatalities N
- UK HSE unacceptable/tolerable/broadly-acceptable boundaries
- Risk integral

### 8.4 Risk Matrix

The risk matrix classifies individual scenarios and aggregates.

**Consequence categories:**
- Catastrophic: >50 fatalities OR >$100M damage
- Major: 10–50 fatalities OR $50–100M
- Moderate: 1–10 fatalities OR $1–50M
- Minor: <1 fatality OR <$1M

**Likelihood categories:**
- Frequent: >10⁻²/year
- Probable: 10⁻²–10⁻³/year
- Occasional: 10⁻³–10⁻⁴/year
- Remote: 10⁻⁴–10⁻⁵/year
- Extremely Remote: <10⁻⁵/year

---

## 9. Advanced Analysis

### 9.1 Batch Processing

Click **Tools → Batch Runner** to run multiple scenarios with
parameter variations. Define parameter sweeps in CSV or table form.

### 9.2 Sensitivity Analysis

Navigate to **Analysis → Sensitivity** to identify which input
parameters most affect results. Uses one-at-a-time (OAT) method.

### 9.3 Monte Carlo Simulation

Navigate to **Analysis → Monte Carlo** for probabilistic analysis.
Define input distributions and number of iterations.

---

## 10. Exporting Results

### 10.1 PDF Report

Click **File → Export Report**:
- Select scenarios to include
- Choose template (Summary/Detailed)
- Configure branding (company logo, header)
- Click **Generate PDF**

### 10.2 Excel Export

Click **File → Export Excel** to export data tables:
- Scenarios and frequencies
- Consequence contours (tabular)
- Risk results

### 10.3 GIS / KML Export

For maps with spatial context:
- **KML**: Overpressure/fire contours for Google Earth
- **GeoJSON**: For GIS integration

### 10.4 Image Export

Right-click any plot → **Save as Image** (PNG, SVG, PDF).

---

## 11. Settings

### 11.1 Units

Configure preferred units:
- **Pressure:** bar, Pa, psi
- **Temperature:** K, °C, °F
- **Distance:** m, ft
- **Mass:** kg, lb

### 11.2 Risk Criteria

Set customized risk acceptance criteria:

| Parameter | Default | Description |
|-----------|---------|-------------|
| IRPA upper limit | 1×10⁻⁴ /year | Maximum tolerable individual risk |
| IRPA lower limit | 1×10⁻⁶ /year | Broadly acceptable |
| FN slope (-1) | 0.01 | Unacceptable FN boundary |
| FN slope (-2) | 1×10⁻⁴ | Acceptable FN boundary |

### 11.3 Appearance

Choose theme (Light/Dark/System), font size, and plot style.

### 11.4 Preferences

- Auto-save interval
- Default project directory
- Report template
- Numerical precision
- Maximum grid resolution

---

## Appendix: Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save |
| `Ctrl+E` | Export Report |
| `Ctrl+R` | Run Active Analysis |
| `F5` | Refresh |
| `F1` | Help |
