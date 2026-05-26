# Rekarisk Methodology Reference

## 1. Introduction

### 1.1 Purpose

Rekarisk is an open-source Quantitative Risk Assessment (QRA) toolkit for
process safety. It implements industry-standard consequence and frequency
models to evaluate process hazards including releases, fires, explosions,
and toxic dispersion.

### 1.2 Scope

The toolkit covers the full QRA pipeline:

1. **Source Term** — mass and energy release rates
2. **Dispersion** — atmospheric transport of hazardous materials
3. **Fire** — thermal radiation from pool, jet, and BLEVE fires
4. **Explosion** — blast overpressure from vapor cloud and confined explosions
5. **Vulnerability** — dose-response relationships (probit functions)
6. **Risk Integration** — event trees, individual risk, societal risk, risk matrix

### 1.3 Standards Referenced

| Standard | Description |
|----------|-------------|
| API 520 | Sizing, Selection, and Installation of Pressure-Relieving Devices |
| API 521 | Pressure-relieving and Depressuring Systems |
| API 752 | Management of Hazards Associated with Location of Process Plant Permanent Buildings |
| CCPS (2010) | Guidelines for Chemical Process Quantitative Risk Analysis, 2nd Ed. |
| CCPS (2000) | Guidelines for Consequence Analysis of Chemical Releases |
| TNO Purple Book | Methods for the Determination of Possible Damage (CPR 16E) |
| TNO Green Book | Methods for the Calculation of Physical Effects (CPR 14E) |
| TNO Yellow Book | Methods for the Calculation of Physical Effects (CPR 14E, 3rd Ed.) |
| ISO 17776:2016 | Petroleum and natural gas industries — Offshore production installations |
| UK HSE | Technical Assessment Guides — Indicative Human Vulnerability |

---

## 2. Source Term Models

### 2.1 Orifice Discharge

**Theory:** Mass flow through a circular orifice driven by pressure difference.

**Liquid discharge (Bernoulli):**

$$\dot{m} = C_d \cdot A \cdot \sqrt{2 \rho \Delta P + 2 \rho^2 g h}$$

where:
- $\dot{m}$ = mass flow rate (kg/s)
- $C_d$ = discharge coefficient (0.62 for sharp-edged orifice)
- $A$ = orifice area ($\pi d^2 / 4$) (m²)
- $\rho$ = liquid density (kg/m³)
- $\Delta P$ = pressure difference across orifice (Pa)
- $h$ = liquid head above orifice (m)

**Gas discharge (choked/subsonic):**

Choked flow criterion:

$$\frac{P_{down}}{P_{up}} \leq \left(\frac{2}{k+1}\right)^{\frac{k}{k-1}}$$

Choked mass flow:

$$\dot{m} = C_d \cdot A \cdot P_{up} \cdot \sqrt{\frac{kM}{RT} \left(\frac{2}{k+1}\right)^{\frac{k+1}{k-1}}}$$

Subsonic mass flow:

$$\dot{m} = C_d \cdot A \cdot P_{up} \cdot \sqrt{\frac{2k}{k-1} \cdot \frac{M}{RT} \left[ r^{2/k} - r^{(k+1)/k} \right]}$$

where $r = P_{down}/P_{up}$.

**Two-phase (Omega method):**

$$\dot{m} = C_d \cdot A \cdot \frac{h_{tg}(T_0) - h_{tg}(T_b)}{v_0^{1/2}} \cdot G_c^*(\omega)$$

**Inputs:**
- Orifice diameter, discharge coefficient
- Upstream/downstream pressure, temperature
- Fluid phase (liquid/gas/two-phase)
- Density, molecular weight, specific heat ratio

**Outputs:**
- Mass flow rate (kg/s)
- Velocity (m/s)
- Flow regime (subsonic/choked)

**Limitations:**
- Assumes sharp-edged orifice
- Steady-state release only
- Homogeneous two-phase flow
- No downstream pressure build-up

### 2.2 Vessel Blowdown (API 521)

**Theory:** Time-dependent depressurization of a vessel through an orifice,
modeling mass and energy balance.

$$\frac{dm}{dt} = -\dot{m}_{out}$$

$$\frac{dU}{dt} = -\dot{m}_{out} \cdot h_{out} + Q_{fire}$$

where:
- $m$ = mass in vessel
- $U$ = internal energy
- $h_{out}$ = enthalpy of escaping fluid
- $Q_{fire}$ = heat input from fire scenario

**Inputs:**
- Vessel volume, wall area
- Initial pressure, temperature
- Orifice diameter, discharge coefficient
- Phase, molecular weight, CP/CV ratio

**Outputs:**
- P(t), T(t), m(t), mdot(t)
- Total mass released (kg)
- Maximum release rate (kg/s)

**Limitations:**
- Lumped parameter (well-mixed vessel)
- Neglects wall thermal inertia in simplified mode
- Two-phase assumes homogeneous flow

### 2.3 Two-Phase Flow

**Flashing fraction:**

$$\chi = \min\left(1, \frac{c_p(T_0 - T_b)}{h_{fg}}\right)$$

**Omega parameter (HEM):**

$$\omega = \frac{x_0 \cdot v_{\text{fg}}}{v_0} + \frac{c_{p,l} \cdot T \cdot P}{\rho_l} \cdot \left(\frac{v_{\text{fg}}}{h_{fg}}\right)^2$$

**Limitations:**
- Homogeneous equilibrium model (HEM) assumes both phases travel at same velocity
- Not suitable for stratified or annular flow regimes

### 2.4 PSV Sizing (API 520)

**Gas/vapor relief area:**

$$A = \frac{W}{C \cdot K_d \cdot K_b \cdot K_c \cdot P_1 \cdot \sqrt{M/(T \cdot Z)}}$$

where $C = 520 \cdot \sqrt{k\left(\frac{2}{k+1}\right)^{\frac{k+1}{k-1}}}$ for choked flow.

**Liquid relief area:**

$$A = \frac{W}{K_d \cdot K_w \cdot \sqrt{2\rho \cdot (P_1 - P_2)}}$$

**Orifice designations:** D through T, following API 526 standard areas.

**Inputs:** Set pressure, flow rate, fluid type, temperature
**Outputs:** Required area (mm²), API designation

### 2.5 Pool Evaporation (Mackay-Matsugu)

**Evaporation rate:**

$$\dot{m}_{\text{evap}} = \frac{k_m \cdot A_{\text{pool}} \cdot M \cdot P_v}{R \cdot T}$$

**Mass transfer coefficient:**

$$k_m = 0.004786 \cdot U^{0.78} \cdot d^{-0.11} \cdot Sc^{-0.67}$$

where $Sc = \nu / D_{AB}$ is the Schmidt number.

**Inputs:** Spill mass, wind speed, surface type, substance properties
**Outputs:** Pool area(t), evaporation rate(t), total evaporated

---

## 3. Dispersion Models

### 3.1 Gaussian Plume (Pasquill-Gifford)

**Concentration at point (x,y,z):**

$$C(x,y,z) = \frac{\dot{m}}{2\pi u \sigma_y \sigma_z} \cdot \exp\left(-\frac{y^2}{2\sigma_y^2}\right) \cdot \left[ \exp\left(-\frac{(z-H)^2}{2\sigma_z^2}\right) + \exp\left(-\frac{(z+H)^2}{2\sigma_z^2}\right) \right]$$

**Dispersion coefficients (sigma):**

$$\sigma_y = a \cdot x^b, \quad \sigma_z = c \cdot x^d$$

where coefficients depend on stability class (A–F) and terrain type.

**Inputs:** Source rate, wind speed, stability class, release height
**Outputs:** 3D or 2D concentration grid, maximum downwind concentrations

**Assumptions:**
- Steady-state, continuous release
- Homogeneous, flat terrain
- Constant wind speed and direction
- No deposition or chemical reactions
- No inversion lid

### 3.2 Gaussian Puff

**Puff concentration:**

$$C(x,y,z,t) = \frac{m}{(2\pi)^{3/2} \sigma_x \sigma_y \sigma_z} \cdot \exp\left(-\frac{(x - ut)^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2}\right) \cdot \left[ \exp\left(-\frac{(z-H)^2}{2\sigma_z^2}\right) + \exp\left(-\frac{(z+H)^2}{2\sigma_z^2}\right) \right]$$

**Assumptions:**
- Instantaneous release
- Puff moves downwind at wind speed
- Sigma values for puff (travel time-based)

### 3.3 Dense Gas (SLAB-type)

**Spreading phase (gravity-driven):**

$$\frac{dR}{dt} = c_{g} \sqrt{g' \cdot h}$$

where $g' = g(\rho_c - \rho_a)/\rho_a$ is reduced gravity.

**Passive transition:** When $g'h / u_*^2 < L_c$ (critical Richardson number).

**Inputs:** Initial mass, radius, height; density ratio
**Outputs:** R(t), H(t), concentration, distance to LFL

**Limitations:**
- Slab model (top-hat profiles)
- Simple energy/entrainment closures
- No terrain effects

---

## 4. Fire Models

### 4.1 Pool Fire (Mudan, Shokri-Beyler)

**Burning rate:**

$$\dot{m}'' = \dot{m}_\infty'' \cdot (1 - e^{-k\beta D})$$

**Flame length (Thomas):**

$$\frac{L}{D} = 42 \cdot \left[ \frac{\dot{m}''}{\rho_a \sqrt{gD}} \right]^{0.61}$$

**Thermal radiation at distance R:**

$$q''(R) = \tau \cdot SEP \cdot F_{view}(R)$$

where SEP is Surface Emissive Power and $F_{view}$ is the geometric view factor.

**Inputs:** Pool diameter, substance (burning rate params), wind speed

### 4.2 Jet Fire (API 521)

**Flame length (Sonju-Hustad):**

$$L = 5.3 \cdot D \cdot \sqrt{\frac{\rho_g}{\rho_a}}$$

or (Kalghatgi):

$$L = D \cdot \frac{\dot{m}}{D} \text{ correlation}$$

**Fraction of heat radiated:** 0.15–0.40, depending on gas type.

### 4.3 BLEVE / Fireball (Roberts, CCPS)

**Fireball diameter:**

$$D = 5.8 \cdot M^{1/3}$$

**Fireball duration:**

$$t_{\text{fb}} = 0.45 \cdot M^{1/3} \quad \text{for } M < 30,000\text{ kg}$$
$$t_{\text{fb}} = 2.6 \cdot M^{1/6} \quad \text{for } M \geq 30,000\text{ kg}$$

**Surface emissive power:**

$$SEP = \frac{\eta \cdot M \cdot \Delta H_c}{\pi D^2 \cdot t_{\text{fb}}}$$

### 4.4 Flash Fire

Flash fire is modeled as flammable cloud extent (LFL boundary). Thermal
consequence is typically handled via probit, but simplified approach
assumes 100% mortality within LFL contour.

---

## 5. Explosion Models

### 5.1 TNT Equivalency (Kingery-Bulmash)

**TNT-equivalent mass:**

$$W_{TNT} = \eta \cdot M \cdot \frac{\Delta H_c}{\Delta H_{c,TNT}}$$

where $\eta$ is the efficiency factor (typically 0.01–0.10).

**Scaled distance:**

$$Z = \frac{R}{W_{TNT}^{1/3}}$$

**Overpressure:** Correlated from Kingery-Bulmash hemispherical surface burst data:

$$\log_{10} P = \sum_{i=0}^{n} c_i \cdot (\log_{10} Z)^i$$

**Inputs:** Mass, heat of combustion, efficiency factor, distance grid
**Outputs:** Overpressure field (Pa), damage contours

**Limitations:**
- Does not account for confinement/congestion
- Overpredicts near-field for VCEs
- Efficiency factor is highly uncertain

### 5.2 TNO Multi-Energy

**Dimensionless scaled distance:**

$$\bar{R} = \frac{R}{(E_c / P_0)^{1/3}}$$

where $E_c = \alpha \cdot M \cdot \Delta H_c$, with $\alpha$ ranging by scenario.

**Blast strength index:** 1 (insignificant) to 10 (detonation)

**Overpressure:** Scaled from TNO curves as function of blast strength.

### 5.3 Baker-Strehlow-Tang (BST)

**Flame speed:** Assigns Mach number based on:
- Fuel reactivity (low/medium/high)
- Congestion (low/medium/high)
- Confinement (1D/2D/3D)

**Scaled distance:**

$$\bar{R} = R \cdot \left(\frac{P_0}{E}\right)^{1/3}$$

**Curves:** Family of blast curves parameterized by Mach number.

---

## 6. Vulnerability Models

### 6.1 Probit Functions

**Probit-to-probability conversion:**

$$P = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{Y-5} e^{-u^2/2} du$$

where $Y$ is the probit value and $P$ is the probability (0–1).

**Thermal probit (Eisenberg):**

$$Y = -14.9 + 2.56 \cdot \ln\left(q^{4/3} \cdot t \cdot 10^{-4}\right)$$

where $q$ is heat flux in W/m² and $t$ is exposure time in seconds.

**Thermal probit (TNO):**

$$Y = -12.8 + 2.56 \cdot \ln\left(q^{4/3} \cdot t\right)$$

**Overpressure probit (Eisenberg — lung damage):**

$$Y = -77.1 + 6.91 \cdot \ln(P_{\text{over}})$$

where $P_{\text{over}}$ is peak overpressure in Pa.

**Overpressure probit (TNO — structural collapse):**

$$Y = -23.8 + 2.92 \cdot \ln(P_{\text{over}})$$

**Toxic probit (general form):**

$$Y = a + b \cdot \ln(C^n \cdot t)$$

where:
- $C$ = concentration (mg/m³)
- $t$ = exposure time (seconds)
- $a, b, n$ = substance-specific constants

### 6.2 Shelter Factor

**Indoor concentration:**

$$C_{\text{in}}(t) = C_{\text{out}} \cdot SF \cdot (1 - e^{-\lambda t})$$

where $SF$ is the shelter factor and $\lambda$ is the air exchange rate.

| Building Type | Shelter Factor | Air Exchange Rate |
|---------------|---------------|-------------------|
| Outdoor       | 1.00          | —                 |
| Sealed        | 0.01–0.05     | 0.1–0.5 h⁻¹       |
| Residential   | 0.10–0.30     | 2–5 h⁻¹           |
| Commercial    | 0.20–0.50     | 5–10 h⁻¹          |

---

## 7. QRA Framework

### 7.1 Event Tree Analysis

Event trees branch from an initiating event through safety barrier
success/failure gates. Each path leads to an outcome scenario with
an associated frequency.

$$F_{\text{scenario}} = f_i \cdot \prod_{j} p_j$$

where $f_i$ is initiating event frequency and $p_j$ are barrier failure/success probabilities.

### 7.2 Individual Risk (IRPA)

**Location-specific IR:**

$$IR(x,y) = \sum_{i} f_i \cdot P_{\text{fatality},i}(x,y)$$

where $f_i$ is scenario frequency and $P_{\text{fatality},i}$ is the conditional
probability of fatality at location (x,y) given scenario $i$.

### 7.3 Societal Risk (FN Curves)

**FN curve:**

$$F(N) = \sum_{i: N_i \geq N} f_i$$

where $F(N)$ is the cumulative frequency of events causing $N$ or more fatalities.

**Risk aversion guidelines** (UK HSE):
- Unacceptable if $F(N) > 0.01/N$ for $N \ge 1$
- Broadly acceptable if $F(N) < 10^{-4}/N$ for $N \ge 1$

### 7.4 Risk Matrix (ISO 17776)

| Likelihood \\ Consequence | Minor | Moderate | Major | Catastrophic |
|---------------------------|-------|----------|-------|-------------|
| Frequent (>10⁻²/y)        | Medium| High     | Extreme| Extreme    |
| Probable (10⁻²–10⁻³/y)    | Medium| Medium   | High  | Extreme    |
| Occasional (10⁻³–10⁻⁴/y)  | Low   | Medium   | Medium| High       |
| Remote (10⁻⁴–10⁻⁵/y)      | Low   | Low      | Medium| Medium     |
| Extremely Remote (<10⁻⁵/y)| Low   | Low      | Low   | Medium     |

---

## 8. Equation of State

### 8.1 Peng-Robinson (PR)

$$P = \frac{RT}{v-b} - \frac{a(T)}{v(v+b) + b(v-b)}$$

$$a(T) = 0.45724 \cdot \frac{R^2 T_c^2}{P_c} \cdot \alpha(T)$$

$$b = 0.07780 \cdot \frac{RT_c}{P_c}$$

$$\alpha(T) = \left[1 + \kappa (1 - \sqrt{T_r})\right]^2$$

$$\kappa = 0.37464 + 1.54226\omega - 0.26992\omega^2$$

Z-factor from cubic: $Z^3 + (B-1)Z^2 + (A-2B-3B^2)Z + (B^2 + B^3 - AB) = 0$

### 8.2 Mixture Rules

**van der Waals mixing rules:**

$$a_m = \sum_i \sum_j x_i x_j \sqrt{a_i a_j} (1 - k_{ij})$$

$$b_m = \sum_i x_i b_i$$

### 8.3 Soave-Redlich-Kwong (SRK)

$$P = \frac{RT}{v-b} - \frac{a(T)}{v(v+b)}$$

Similar structure to PR but different constants and $\alpha$ function.

---

## 9. References

1. CCPS, *Guidelines for Chemical Process Quantitative Risk Analysis*, 2nd Ed., AIChE, 2010.
2. CCPS, *Guidelines for Consequence Analysis of Chemical Releases*, AIChE, 2000.
3. TNO, *Methods for the Calculation of Physical Effects* (Yellow Book), CPR 14E, 3rd Ed., 2005.
4. TNO, *Methods for the Determination of Possible Damage* (Purple Book), CPR 16E, 2005.
5. API Standard 520, *Sizing, Selection, and Installation of Pressure-Relieving Devices*, Part 1, 9th Ed., 2014.
6. API Standard 521, *Pressure-relieving and Depressuring Systems*, 7th Ed., 2020.
7. Kingery, C.N. and Bulmash, G., *Airblast Parameters from TNT Spherical Air Burst and Hemispherical Surface Burst*, ARBRL-TR-02555, 1984.
8. Baker, Q.A., Tang, M.J., Scheier, E.A., Silva, G.J., *Vapor Cloud Explosion Analysis*, Process Safety Progress 15(2), 106–109, 1996.
9. Mudan, K.S., *Thermal Radiation Hazards from Hydrocarbon Pool Fires*, Prog. Energy Combust. Sci., 1984.
10. Roberts, A.F., *Thermal Radiation Hazards from Releases of LPG from Pressurised Storage*, Fire Safety J., 1981/82.
11. ISO 17776:2016, *Petroleum and Natural Gas Industries — Offshore Production Installations*.
12. Mackay, D. and Matsugu, R.S., *Evaporation Rates of Liquid Hydrocarbon Spills on Land and Water*, Can. J. Chem. Eng., 1973.
13. Peng, D.Y. and Robinson, D.B., *A New Two-Constant Equation of State*, Ind. Eng. Chem. Fundam., 1976.
14. Soave, G., *Equilibrium Constants from a Modified Redlich-Kwong Equation of State*, Chem. Eng. Sci., 1972.
