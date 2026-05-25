# Rekarisk

**Consequence & Risk Analysis for Safety Engineers**

Desktop software for dispersion, fire, explosion modeling, and Quantitative Risk Assessment (QRA). Built to be a capable open-source alternative to proprietary tools.

## Status

🔧 **Phase 0: Repository Setup** — Scaffolding and CI/CD

## Features (Planned)

| Module | Status |
|--------|--------|
| Source Term / Discharge Engine | ⬜ Planned |
| Dispersion (Gaussian, Dense Gas) | ⬜ Planned |
| Fire (Pool, Jet, BLEVE, Flash) | ⬜ Planned |
| Explosion (TNT, TNO, BST) | ⬜ Planned |
| 3D Terrain & Obstacles | ⬜ Planned |
| QRA Framework | ⬜ Planned |
| Toxicology & Vulnerability | ⬜ Planned |
| Batch / Sensitivity / Monte Carlo | ⬜ Planned |
| PDF Reports & GIS Export | ⬜ Planned |

## Tech Stack

- **Python 3.11+** with PyQt6
- NumPy, SciPy, Matplotlib, Cartopy
- Custom Equation of State (Peng-Robinson, SRK)
- ReportLab (PDF), openpyxl (Excel)

## Documentation

- [Design Plan](docs/CONSEQUENCE_ANALYSIS_PLAN.md)
- [Installation Guide](docs/INSTALLATION.md)
- [User Manual](docs/USER_MANUAL.md)
- [Tutorial](docs/TUTORIAL.md)
- [Test Cases](docs/TEST_CASES.md)
- [Methodology Reference](docs/METHODOLOGY.md)

## Quick Start

```bash
# Clone
git clone https://github.com/arienug20/rekarisk.git
cd rekarisk

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Launch (when ready)
python -m rekarisk
```

## License

TBD — GPLv3 or MIT

---

*Rekarisk is under active development. See the [Design Plan](docs/CONSEQUENCE_ANALYSIS_PLAN.md) for the full roadmap.*
