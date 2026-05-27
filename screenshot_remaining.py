#!/usr/bin/env python3
"""Capture remaining panels."""
import os, sys
os.environ.setdefault("DISPLAY", ":99")
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

app = QApplication(sys.argv)
OUTPUT = "/home/arienugraha-rei/rekarisk/screenshots"

def cap(w, n):
    p = w.grab()
    p.save(os.path.join(OUTPUT, f"{n}.png"))
    print(f"  ✓ {n}.png")
    w.close()
    app.processEvents()

def run():
    from rekarisk.core.substance_db import get_database
    db = get_database()

    # Report Dialog
    from rekarisk.ui.report_dialog import ReportDialog
    d = ReportDialog(project_data={"name": "Demo"}, results=[])
    d.resize(800, 600); d.show(); app.processEvents()
    cap(d, "10_report_dialog")

    # Sensitivity
    from rekarisk.ui.sensitivity_dialog import SensitivityDialog
    from rekarisk.models.dispersion.gaussian_plume import calculate_plume, PlumeInput
    d = SensitivityDialog(lambda p: calculate_plume(PlumeInput(**p)), {"source_rate": 1.0, "wind_speed": 3.0}, "max_concentration")
    d.resize(900, 700); d.show(); app.processEvents()
    cap(d, "11_sensitivity_dialog")

    # Monte Carlo
    from rekarisk.ui.monte_carlo_dialog import MonteCarloDialog
    from rekarisk.analysis.monte_carlo import Uniform
    d = MonteCarloDialog(lambda p: calculate_plume(PlumeInput(**p)), {"source_rate": Uniform(0.5, 5.0)}, ["max_concentration"])
    d.resize(900, 700); d.show(); app.processEvents()
    cap(d, "12_monte_carlo_dialog")

    # Batch Runner
    from rekarisk.ui.batch_dialog import BatchDialog
    d = BatchDialog(lambda p: calculate_plume(PlumeInput(**p)), {"Light": {"source_rate": 0.5}}, [{"label": "D", "wind_speed": 3.0}])
    d.resize(900, 700); d.show(); app.processEvents()
    cap(d, "13_batch_runner_dialog")

    # Case Comparison
    from rekarisk.ui.case_comparison import CaseComparisonPanel
    w = CaseComparisonPanel(); w.resize(900, 600); w.show(); app.processEvents()
    cap(w, "14_case_comparison")

    # Terrain
    from rekarisk.ui.terrain_dialog import ObstacleTab
    w = ObstacleTab(); w.resize(800, 500); w.show(); app.processEvents()
    cap(w, "15_terrain_obstacles")

    # Mixture Editor
    from rekarisk.ui.mixture_editor import MixtureEditorDialog
    d = MixtureEditorDialog(db=db); d.resize(700, 500); d.show(); app.processEvents()
    cap(d, "16_mixture_editor")

    print(f"\n✅ Done!")
    app.quit()

QTimer.singleShot(500, run)
app.exec()
