# Rekarisk

<p align="center">
  <img src="docs/logo.svg" alt="Rekarisk" width="200">
</p>

<p align="center">
  <strong>Consequence &amp; Risk Analysis for Safety Engineers</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/version-1.0.0--dev-orange.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/arienug20/rekarisk"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform"></a>
</p>

---

## Overview

**Rekarisk** is a desktop application for consequence analysis and quantitative risk assessment (QRA) of hazardous chemical releases. It models source term discharge, atmospheric dispersion, fires, explosions, and toxicological impacts—integrating everything into a unified QRA framework with risk ranking, FN curves, and GIS-capable reporting.

Built as an open-source alternative to proprietary safety-engineering tools, Rekarisk is aimed at process safety engineers, HSE professionals, researchers, and students.

## Features

| Module | Capabilities |
|--------|-------------|
| **Source Term** | Orifice discharge, pipe flow, two-phase release, relief valve sizing, pool evaporation, vessel depressurisation |
| **Equation of State** | Peng-Robinson, Soave-Redlich-Kwong, VLE flash, hydrate curve, phase envelope |
| **Meteorology** | Pasquill-Gifford stability class, wind-profile power law, wind rose |
| **Dispersion** | Gaussian plume (continuous), Gaussian puff (instantaneous), dense-gas (SLAB-style) |
| **Fire** | Pool fire, jet fire, BLEVE, flash fire — thermal radiation contours |
| **Explosion** | TNT equivalency, TNO Multi-Energy, Baker-Strehlow-Tang |
| **Terrain** | Obstacle definitions, line-of-sight engine, DEM loading |
| **Toxicology** | Probit vulnerability, toxic dose, shelter factors |
| **QRA** | Event tree, failure frequency, ignition probability, individual risk (IR), societal risk (FN), risk matrix |
| **Advanced Analysis** | Batch runner, sensitivity analysis, Monte Carlo simulation, worst-case search |
| **Reporting** | PDF reports, Excel export, GIS (GeoJSON/KML), image export, text reports |
| **Audit Trail** | Session checkpoints, versioned project files (`.caproj`) |
| **Visualization** | Contour plots, iso-lines, wind rose, risk maps |

## Quick Start

### Prerequisites

- **Python 3.11 or later** (3.12 supported)
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/arienug20/rekarisk.git
cd rekarisk

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install Rekarisk and its dependencies
pip install -e ".[dev]"
```

### Launch

```bash
# GUI
python -m rekarisk

# CLI
rekarisk --help
rekarisk version
rekarisk substances methane
rekarisk run examples/scenario.json -o results.json
```

## Usage Examples

### Run a Scenario from the CLI

```json
{
  "type": "dispersion",
  "substance": "methane",
  "release_rate_kg_s": 5.0,
  "wind_speed_m_s": 3.5,
  "stability_class": "D",
  "ambient_temp_K": 298.15
}
```

```bash
rekarisk run scenario.json -f json -o output.json
```

### Batch Processing

```bash
rekarisk batch batch_config.json
```

### Search the Substance Database

```bash
$ rekarisk substances methane
Matches for 'methane':
  • Methane  CH₄  (74-82-8)
  • Methanol  CH₃OH  (67-56-1)
```

## Project Structure

```
rekarisk/
├── src/rekarisk/
│   ├── __init__.py              # Package root
│   ├── __main__.py              # python -m entry point
│   ├── __version__.py           # Version management
│   ├── cli.py                   # CLI (rekarisk command)
│   ├── core/                    # EoS, DIPPR, substance DB, units, validation
│   ├── meteorology/             # P-G stability, wind profile, weather data
│   ├── models/
│   │   ├── source_term/         # Orifice, pipe, two-phase, relief, pool, vessel
│   │   ├── dispersion/          # Gaussian plume/puff, dense gas
│   │   ├── fire/                # Pool fire, jet fire, BLEVE, flash fire
│   │   ├── explosion/           # TNT, TNO Multi-Energy, BST
│   │   ├── vulnerability/       # Probit, toxic dose, shelter
│   │   └── qra/                 # Event tree, IR, FN, risk matrix
│   ├── analysis/                # Batch, sensitivity, Monte Carlo, worst-case
│   ├── terrain/                 # Obstacles, LOS, DEM
│   ├── report/                  # PDF, Excel, GIS, image, text exports
│   ├── ui/                      # PyQt6 GUI panels and dialogs
│   └── visualization/           # Contours, iso-lines, risk maps
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Substance database, example data
├── pyproject.toml               # Build configuration
├── requirements.txt             # Pip dependencies
├── CHANGELOG.md                 # Version history
└── LICENSE                      # MIT License
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a branch for your feature.
2. Follow [Conventional Commits](https://www.conventionalcommits.org/).
3. Add tests for new functionality.
4. Run `pytest` and `ruff check` before submitting.
5. Open a pull request against `main`.

For larger features or changes, please [open an issue](https://github.com/arienug20/rekarisk/issues) first to discuss.

### Development Setup

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## License

Rekarisk is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

## References &amp; Credits

Rekarisk builds on decades of published models and guidance—the work of hundreds of researchers and practitioners. Key references include:

| Reference | Source |
|-----------|--------|
| **TNO Yellow Book** — Methods for the calculation of physical effects | TNO, 1997 / 2005 |
| **TNO Green Book** — Methods for the determination of possible damage | TNO, 1992 |
| **TNO Purple Book** — Guidelines for quantitative risk assessment | TNO, 2005 |
| **CCPS Guidelines** for Consequence Analysis, QRA, and VCE | CCPS / AIChE |
| **HSE** — Failure Rate &amp; Event Data (FRED), risk criteria | UK HSE |
| **API RP 521** — Pressure-relieving and depressuring systems | API |
| **Crowl &amp; Louvar** — Chemical Process Safety: Fundamentals with Applications | Pearson |
| **Lees' Loss Prevention** in the Process Industries | Butterworth-Heinemann |
| **AIChE/CCPS** — Guidelines for Chemical Process Quantitative Risk Analysis (CPQRA) | CCPS |

---

<p align="center">
  <em>Rekarisk — safety by design.</em>
</p>
