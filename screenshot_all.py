#!/usr/bin/env python3
"""Screenshot all Rekarisk frontend panels and dialogs."""

import os
import sys

os.environ.setdefault("DISPLAY", ":99")
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt6.QtWidgets import QApplication, QTabWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QScreen

app = QApplication(sys.argv)

OUTPUT_DIR = "/home/arienugraha-rei/rekarisk/screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track all windows
captures = []

def capture_widget(widget, name):
    """Grab widget and save."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    pixmap = widget.grab()
    pixmap.save(path)
    print(f"  ✓ Saved {name}.png ({pixmap.width()}x{pixmap.height()})")
    captures.append(path)

def run_captures():
    """Capture all panels."""
    from rekarisk.core.substance_db import get_database
    db = get_database()

    # ═══════════════════════════════════════════════════════════
    # 1. Main Window
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Main Window...")
    from rekarisk.ui.main_window import MainWindow
    mw = MainWindow()
    mw.resize(1400, 900)
    mw.show()
    app.processEvents()
    capture_widget(mw, "01_main_window")

    # ═══════════════════════════════════════════════════════════
    # 2. Source Term Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Source Term Panel...")
    from rekarisk.ui.source_term_panel import SourceTermPanel
    st_panel = SourceTermPanel()
    st_panel.resize(800, 700)
    st_panel.show()
    app.processEvents()

    # Screenshot each sub-tab
    tabs = st_panel.tabs
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        app.processEvents()
        tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
        capture_widget(st_panel, f"02_source_term_{tab_name}")

    st_panel.close()

    # ═══════════════════════════════════════════════════════════
    # 3. Dispersion Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Dispersion Panel...")
    from rekarisk.ui.dispersion_panel import DispersionPanel
    disp_panel = DispersionPanel()
    disp_panel.resize(800, 700)
    disp_panel.show()
    app.processEvents()

    tabs = disp_panel._tabs if hasattr(disp_panel, '_tabs') else disp_panel.tabs if hasattr(disp_panel, 'tabs') else None
    if tabs:
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
            capture_widget(disp_panel, f"03_dispersion_{tab_name}")
    else:
        capture_widget(disp_panel, "03_dispersion_panel")

    disp_panel.close()

    # ═══════════════════════════════════════════════════════════
    # 4. Fire Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Fire Panel...")
    from rekarisk.ui.fire_panel import FirePanel
    fire_panel = FirePanel()
    fire_panel.resize(800, 700)
    fire_panel.show()
    app.processEvents()

    tabs = fire_panel._tabs if hasattr(fire_panel, '_tabs') else fire_panel.tabs if hasattr(fire_panel, 'tabs') else None
    if tabs:
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
            capture_widget(fire_panel, f"04_fire_{tab_name}")
    else:
        capture_widget(fire_panel, "04_fire_panel")

    fire_panel.close()

    # ═══════════════════════════════════════════════════════════
    # 5. Explosion Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Explosion Panel...")
    from rekarisk.ui.explosion_panel import ExplosionPanel
    expl_panel = ExplosionPanel()
    expl_panel.resize(800, 700)
    expl_panel.show()
    app.processEvents()

    tabs = expl_panel._tabs if hasattr(expl_panel, '_tabs') else expl_panel.tabs if hasattr(expl_panel, 'tabs') else None
    if tabs:
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
            capture_widget(expl_panel, f"05_explosion_{tab_name}")
    else:
        capture_widget(expl_panel, "05_explosion_panel")

    expl_panel.close()

    # ═══════════════════════════════════════════════════════════
    # 6. Vulnerability Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Vulnerability Panel...")
    from rekarisk.ui.vulnerability_panel import VulnerabilityPanel
    vuln_panel = VulnerabilityPanel()
    vuln_panel.resize(800, 700)
    vuln_panel.show()
    app.processEvents()

    tabs = vuln_panel._tabs if hasattr(vuln_panel, '_tabs') else vuln_panel.tabs if hasattr(vuln_panel, 'tabs') else None
    if tabs:
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
            capture_widget(vuln_panel, f"06_vulnerability_{tab_name}")
    else:
        capture_widget(vuln_panel, "06_vulnerability_panel")

    vuln_panel.close()

    # ═══════════════════════════════════════════════════════════
    # 7. QRA Panel
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing QRA Panel...")
    from rekarisk.ui.qra_panel import QRAPanel
    try:
        qra_panel = QRAPanel()
        qra_panel.resize(800, 700)
        qra_panel.show()
        app.processEvents()

        tabs = qra_panel._tabs if hasattr(qra_panel, '_tabs') else qra_panel.tabs if hasattr(qra_panel, 'tabs') else None
        if tabs:
            for i in range(tabs.count()):
                tabs.setCurrentIndex(i)
                app.processEvents()
                tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_")
                capture_widget(qra_panel, f"07_qra_{tab_name}")
        else:
            capture_widget(qra_panel, "07_qra_panel")

        qra_panel.close()
    except Exception as e:
        print(f"  ⚠️ QRA Panel error: {e}")

    # ═══════════════════════════════════════════════════════════
    # 8. Weather Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Weather Dialog...")
    from rekarisk.ui.weather_dialog import WeatherDialog
    weather_dlg = WeatherDialog()
    weather_dlg.resize(900, 700)
    weather_dlg.show()
    app.processEvents()

    for i in range(weather_dlg.tabs.count()):
        weather_dlg.tabs.setCurrentIndex(i)
        app.processEvents()
        tab_name = weather_dlg.tabs.tabText(i).replace(" ", "_").replace("/", "_")
        capture_widget(weather_dlg, f"08_weather_{tab_name}")

    weather_dlg.close()

    # ═══════════════════════════════════════════════════════════
    # 9. Substance Selector
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Substance Selector...")
    from rekarisk.ui.substance_selector import SubstanceSelector
    sub_sel = SubstanceSelector(db)
    sub_sel.resize(400, 600)
    sub_sel.show()
    app.processEvents()
    capture_widget(sub_sel, "09_substance_selector")
    sub_sel.close()

    # ═══════════════════════════════════════════════════════════
    # 10. Report Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Report Dialog...")
    from rekarisk.ui.report_dialog import ReportDialog
    report_dlg = ReportDialog(project_data={"name": "Demo Project"}, results=[])
    report_dlg.resize(800, 600)
    report_dlg.show()
    app.processEvents()
    capture_widget(report_dlg, "10_report_dialog")
    report_dlg.close()

    # ═══════════════════════════════════════════════════════════
    # 11. Sensitivity Analysis Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Sensitivity Dialog...")
    from rekarisk.ui.sensitivity_dialog import SensitivityDialog
    from rekarisk.models.dispersion.gaussian_plume import calculate_plume, PlumeInput
    def dummy_model(params):
        return calculate_plume(PlumeInput(**params))
    sens_dlg = SensitivityDialog(
        model_function=dummy_model,
        base_params={"source_rate": 1.0, "wind_speed": 3.0, "stability_class": "D"},
        output_key="max_concentration",
    )
    sens_dlg.resize(900, 700)
    sens_dlg.show()
    app.processEvents()
    capture_widget(sens_dlg, "11_sensitivity_dialog")
    sens_dlg.close()

    # ═══════════════════════════════════════════════════════════
    # 12. Monte Carlo Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Monte Carlo Dialog...")
    from rekarisk.ui.monte_carlo_dialog import MonteCarloDialog
    from rekarisk.analysis.monte_carlo import Uniform, Normal
    mc_dlg = MonteCarloDialog(
        model_function=dummy_model,
        parameters={"source_rate": Uniform(0.5, 5.0), "wind_speed": Uniform(1.0, 8.0)},
        output_keys=["max_concentration"],
    )
    mc_dlg.resize(900, 700)
    mc_dlg.show()
    app.processEvents()
    capture_widget(mc_dlg, "12_monte_carlo_dialog")
    mc_dlg.close()

    # ═══════════════════════════════════════════════════════════
    # 13. Batch Runner Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Batch Runner Dialog...")
    from rekarisk.ui.batch_dialog import BatchDialog
    batch_dlg = BatchDialog(
        model_function=dummy_model,
        scenario_templates={"Light Leak": {"source_rate": 0.5}, "Major Release": {"source_rate": 5.0}},
        weather_options=[{"label": "D - Neutral", "wind_speed": 3.0, "stability": "D"}],
    )
    batch_dlg.resize(900, 700)
    batch_dlg.show()
    app.processEvents()
    capture_widget(batch_dlg, "13_batch_runner_dialog")
    batch_dlg.close()

    # ═══════════════════════════════════════════════════════════
    # 14. Case Comparison
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Case Comparison...")
    from rekarisk.ui.case_comparison import CaseComparisonPanel
    case_cmp = CaseComparisonPanel()
    case_cmp.resize(900, 600)
    case_cmp.show()
    app.processEvents()
    capture_widget(case_cmp, "14_case_comparison")
    case_cmp.close()

    # ═══════════════════════════════════════════════════════════
    # 15. Terrain Dialog
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Terrain Dialog...")
    from rekarisk.ui.terrain_dialog import ObstacleTab
    terrain_tab = ObstacleTab()
    terrain_tab.resize(800, 500)
    terrain_tab.show()
    app.processEvents()
    capture_widget(terrain_tab, "15_terrain_obstacles")
    terrain_tab.close()

    # ═══════════════════════════════════════════════════════════
    # 16. Mixture Editor
    # ═══════════════════════════════════════════════════════════
    print("\n📸 Capturing Mixture Editor...")
    from rekarisk.ui.mixture_editor import MixtureEditorDialog
    mix_dlg = MixtureEditorDialog(db=db)
    mix_dlg.resize(700, 500)
    mix_dlg.show()
    app.processEvents()
    capture_widget(mix_dlg, "16_mixture_editor")
    mix_dlg.close()

    # Done
    mw.close()
    print(f"\n✅ Total screenshots: {len(captures)}")
    print(f"📁 Saved to: {OUTPUT_DIR}")

    app.quit()

# Run with a small delay to let Xvfb settle
QTimer.singleShot(500, run_captures)
app.exec()
