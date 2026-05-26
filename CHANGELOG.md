# Changelog

All notable changes to Rekarisk are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-dev] — 2026-05-26

### Added

- **Phase 1 — Foundation:** Core modules (constants, units, validation, substance database, DIPPR property estimation), UI skeleton with main window and dock-based layout.
- **Phase 2 — Source Term Engine:** Six discharge models — orifice, pipe flow, two-phase release, relief valve sizing, pool evaporation, vessel depressurisation. Includes plotting utilities and interactive panel.
- **Phase 3 — Equation of State:** Peng-Robinson, Soave-Redlich-Kwong, VLE flash, hydrate curve prediction, and phase envelope computation.
- **Phase 4 — Meteorology:** Pasquill-Gifford stability classification (rural/urban), wind-profile power-law exponent, wind rose generation, weather data handling.
- **Phase 5 — Dispersion:** Gaussian plume model (continuous release), Gaussian puff model (instantaneous release), dense-gas dispersion model (SLAB-style), dispersion dispatcher for auto-selection.
- **Phase 6 — Fire:** Pool fire (radiative fraction, view factor), jet fire (API/Shell model), BLEVE (fireball, peak emissive power), flash fire (LFL contour). Thermal radiation result panels.
- **Phase 7 — Explosion:** TNT equivalency model, TNO Multi-Energy model, Baker-Strehlow-Tang model. Overpressure curve and impulse computation.
- **Phase 8 — Terrain:** Obstacle definitions (box/wall/cylinder), line-of-sight engine, DEM loading from GeoTIFF and SRTM HGT.
- **Phase 9 — Toxicology & Vulnerability:** Probit functions for toxicity/thermal/pressure, toxic dose calculation, shelter factor model, vulnerability calculator.
- **Phase 10 — QRA:** Event tree analysis, failure frequency database, ignition probability model, individual risk (IR) mapping, societal risk (FN curve), risk matrix (e.g., HSE UK tolerability).
- **Phase 11 — Advanced Analysis:** Batch runner, sensitivity analysis (one-at-a-time), Monte Carlo simulation, worst-case search.
- **Phase 12 — Reporting:** PDF report generator (ReportLab), Excel export (openpyxl), GIS export (GeoJSON, GeoPackage), image export (Matplotlib), text report export.
- **Phase 13 — Audit Trail & Project Files:** Checkpoint system, auto-save, versioned `.caproj` project files (JSON-based ZIP), audit viewer in the GUI.
- **Phase 14 — Tests:** Unit tests for core modules, source term, EoS, meteorology, dispersion, fire, explosion, vulnerability, and QRA. Benchmark harness.
- **Phase 15 — Documentation:** Design plan, installation guide, user manual, tutorial, test cases reference, methodology reference.
- **Phase 16 — Visualization:** Contour plotting, iso-line generation, wind rose charts, FN curve rendering, risk maps.
- **Phase 17 — Packaging & Deployment:** Version management (`__version__.py`), CLI (`cli.py`), `pyproject.toml` packaging, `python -m rekarisk` entry point, README, LICENSE, CHANGELOG, `.gitignore`.
