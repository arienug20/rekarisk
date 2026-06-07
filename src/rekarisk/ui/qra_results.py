"""
Rekarisk UI — QRA Results Display.

Quantitative Risk Assessment results:
  - Individual Risk (IR) contour plot
  - FN curve (log-log) with criterion lines
  - Risk matrix display (colour-coded grid)
  - Summary table: total IR, FN values, ALARP status
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QFormLayout, QTextEdit, QFileDialog, QMessageBox, QSplitter,
    QHeaderView, QFrame, QDoubleSpinBox, QSpinBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..models.qra.individual_risk import (
    IndividualRiskResult, calculate_ir_grid, get_ir_at_source,
    get_ir_at_distance, RISK_THRESHOLDS,
)
from ..models.qra.societal_risk import (
    FNData, FNCriterion, calculate_fn_curve, fn_data_to_plot,
    compare_to_criterion, FN_CRITERIA, FNStatus,
)
from ..models.qra.risk_matrix import (
    risk_level, classify_likelihood, classify_consequence,
    risk_matrix_table, risk_matrix_html, RiskLevel,
    LikelihoodLevel, ConsequenceLevel, DEFAULT_MATRIX,
)
from ..models.qra.failure_frequency import classify_frequency, FrequencyClass


# ── Helper: Matplotlib Canvas ─────────────────────────────────────────

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt6."""

    def __init__(self, width: int = 6, height: int = 4, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


# ── IR Contour Tab ────────────────────────────────────────────────────

class IRContourTab(QWidget):
    """Individual Risk contour plot and grid display."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._result: Optional[IndividualRiskResult] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Matplotlib canvas for contour plot
        self._canvas = MplCanvas(width=6, height=4, dpi=100)
        layout.addWidget(self._canvas)

        # Controls
        ctrl_layout = QHBoxLayout()

        self._threshold_combo = QComboBox()
        self._threshold_combo.addItems([
            "1e-3 /yr (Intolerable)", "1e-4 /yr (Tolerable)",
            "1e-5 /yr (Acceptable)", "1e-6 /yr (Negligible)"
        ])
        ctrl_layout.addWidget(QLabel("Contour Level:"))
        ctrl_layout.addWidget(self._threshold_combo)

        refresh_btn = QPushButton("🔄 Refresh Plot")
        refresh_btn.clicked.connect(self._refresh_plot)
        ctrl_layout.addWidget(refresh_btn)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Summary table
        sum_group = QGroupBox("IR Summary")
        sum_layout = QFormLayout(sum_group)

        self._max_ir_label = QLabel("—")
        sum_layout.addRow("Maximum IR:", self._max_ir_label)

        self._ir_at_source_label = QLabel("—")
        sum_layout.addRow("IR at Source:", self._ir_at_source_label)

        self._ir_100m_label = QLabel("—")
        sum_layout.addRow("IR at 100 m:", self._ir_100m_label)

        self._ir_500m_label = QLabel("—")
        sum_layout.addRow("IR at 500 m:", self._ir_500m_label)

        layout.addWidget(sum_group)

        # Threshold distances
        thresh_group = QGroupBox("Threshold Distances")
        self._thresh_table = QTableWidget(0, 2)
        self._thresh_table.setHorizontalHeaderLabels(["Threshold", "Distance (m)"])
        self._thresh_table.horizontalHeader().setStretchLastSection(True)
        thresh_layout = QVBoxLayout(thresh_group)
        thresh_layout.addWidget(self._thresh_table)

        layout.addWidget(thresh_group)

    def set_result(self, result: IndividualRiskResult) -> None:
        """Set the IR result and refresh display."""
        self._result = result
        self._refresh_plot()
        self._refresh_summary()

    def _refresh_plot(self) -> None:
        """Render the IR contour plot."""
        if self._result is None:
            return

        self._canvas.fig.clear()
        ax = self._canvas.fig.add_subplot(111)

        X, Y = np.meshgrid(self._result.x_coords, self._result.y_coords)
        Z = self._result.ir_grid

        # Log scale for IR values
        Z_plot = np.where(Z > 0, Z, 1e-12)

        # Contour fill
        levels = [1e-6, 1e-5, 1e-4, 1e-3]
        contour = ax.contourf(
            X, Y, Z_plot,
            levels=levels,
            colors=['#E8F5E9', '#C8E6C9', '#FFF9C4', '#FFCDD2'],
            extend='both',
        )
        ax.contour(X, Y, Z_plot, levels=levels, colors='black', linewidths=0.5)

        # Colorbar
        cbar = self._canvas.fig.colorbar(contour, ax=ax, label='IR (per year)')
        cbar.set_ticks(levels)
        cbar.set_ticklabels(['1e-6', '1e-5', '1e-4', '1e-3'])

        # Source marker
        ax.plot(0, 0, 'r*', markersize=12, label='Source')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Individual Risk Contours')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()

    def _refresh_summary(self) -> None:
        """Refresh IR summary labels and threshold table."""
        if self._result is None:
            return

        max_ir = self._result.max_ir
        self._max_ir_label.setText(f"{max_ir:.4e} /yr")

        ir_source = get_ir_at_source(self._result)
        self._ir_at_source_label.setText(f"{ir_source:.4e} /yr")

        ir_100 = get_ir_at_distance(self._result, 100.0)
        self._ir_100m_label.setText(f"{ir_100:.4e} /yr")

        ir_500 = get_ir_at_distance(self._result, 500.0)
        self._ir_500m_label.setText(f"{ir_500:.4e} /yr")

        # Threshold distances table
        self._thresh_table.setRowCount(len(self._result.threshold_distances))
        for i, (thresh, dist) in enumerate(
            sorted(self._result.threshold_distances.items())
        ):
            self._thresh_table.setItem(i, 0, QTableWidgetItem(thresh))
            dist_item = QTableWidgetItem(f"{dist:.1f}")
            dist_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._thresh_table.setItem(i, 1, dist_item)

    def set_lsir_data(self, lsir: dict) -> None:
        """Display LSIR data from pipeline as a table."""
        if not lsir:
            return
        # Build a simple text summary
        vals = sorted(lsir.values(), reverse=True)
        max_ir = max(vals) if vals else 0
        self._max_ir_label.setText(f"{max_ir:.4e} /yr")

        # Find nearest receptor distances
        for label_target, d_target in [("source", 0), ("100m", 100), ("500m", 500)]:
            best_val = 0
            for key, val in lsir.items():
                # Parse location from key like "(x,y)"
                try:
                    import re
                    nums = re.findall(r'-?[\d.]+', str(key))
                    if len(nums) >= 1:
                        d = abs(float(nums[0]))
                        if abs(d - d_target) < 30:
                            best_val = max(best_val, val)
                except (ValueError, IndexError):
                    pass
            if label_target == "source":
                self._ir_at_source_label.setText(f"{best_val:.4e} /yr")
            elif label_target == "100m":
                self._ir_100m_label.setText(f"{best_val:.4e} /yr")
            elif label_target == "500m":
                self._ir_500m_label.setText(f"{best_val:.4e} /yr")

        # Threshold distances table from LSIR data
        thresholds = [
            ("1e-3 /yr", 1e-3),
            ("1e-4 /yr", 1e-4),
            ("1e-5 /yr", 1e-5),
            ("1e-6 /yr", 1e-6),
        ]
        self._thresh_table.setRowCount(len(thresholds))
        for i, (label, thresh) in enumerate(thresholds):
            self._thresh_table.setItem(i, 0, QTableWidgetItem(label))
            # Count locations exceeding threshold
            n_exceed = sum(1 for v in vals if v >= thresh)
            item = QTableWidgetItem(f"{n_exceed} locations")
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._thresh_table.setItem(i, 1, item)


# ── FN Curve Tab ──────────────────────────────────────────────────────

class FNCurveTab(QWidget):
    """FN curve (log-log) plot with criterion lines."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fn_data: Optional[FNData] = None
        self._criterion: Optional[FNCriterion] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Canvas
        self._canvas = MplCanvas(width=6, height=4, dpi=100)
        layout.addWidget(self._canvas)

        # Controls
        ctrl_layout = QHBoxLayout()

        self._criterion_combo = QComboBox()
        for key in FN_CRITERIA:
            self._criterion_combo.addItem(key.replace("_", " ").title(), key)
        self._criterion_combo.currentIndexChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(QLabel("Criterion:"))
        ctrl_layout.addWidget(self._criterion_combo)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_plot)
        ctrl_layout.addWidget(refresh_btn)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Summary
        sum_group = QGroupBox("FN & PLL Summary")
        sum_layout = QFormLayout(sum_group)

        self._pll_label = QLabel("—")
        sum_layout.addRow("PLL (Potential Loss of Life):", self._pll_label)

        self._max_n_label = QLabel("—")
        sum_layout.addRow("Max Fatalities (single event):", self._max_n_label)

        self._total_freq_label = QLabel("—")
        sum_layout.addRow("Total Accident Frequency:", self._total_freq_label)

        self._exp_fat_label = QLabel("—")
        sum_layout.addRow("Expected Fatalities/yr:", self._exp_fat_label)

        self._alarp_status_label = QLabel("—")
        sum_layout.addRow("ALARP Status:", self._alarp_status_label)

        layout.addWidget(sum_group)

    def set_result(
        self,
        fn_data: FNData,
        criterion: Optional[FNCriterion] = None,
    ) -> None:
        """Set FN data and criterion."""
        self._fn_data = fn_data
        self._criterion = criterion
        self._refresh_plot()
        self._refresh_summary()

    def set_pipeline_fn(self, fn: dict) -> None:
        """Set FN data from pipeline output dict.

        Args:
            fn: Dict with 'n' (list of fatalities) and 'f' (cumulative freq).
        """
        n_vals = fn.get("n", [])
        f_vals = fn.get("f", [])
        if not n_vals:
            return

        # Create FNData-compatible object for plotting
        try:
            self._fn_data = FNData(
                n_values=np.array(n_vals, dtype=float),
                f_values=np.array(f_vals, dtype=float),
                potential_loss_of_life=0,
                max_n=max(n_vals) if n_vals else 0,
                total_frequency=sum(f_vals) if f_vals else 0,
                expected_fatalities=0,
                alarp_status={},
            )
        except Exception:
            # Fallback: store raw data
            self._fn_data = type("obj", (object,), {
                "n_values": np.array(n_vals, dtype=float),
                "f_values": np.array(f_vals, dtype=float),
                "max_n": max(n_vals) if n_vals else 0,
                "total_frequency": sum(f_vals) if f_vals else 0,
                "expected_fatalities": 0,
                "potential_loss_of_life": 0,
                "alarp_status": {},
            })()

        self._refresh_plot()
        self._refresh_summary()

    def _refresh_plot(self) -> None:
        """Render the FN curve."""
        if self._fn_data is None:
            return

        self._canvas.fig.clear()
        ax = self._canvas.fig.add_subplot(111)

        n_vals, f_vals = self._fn_data.n_values, self._fn_data.f_values

        if len(n_vals) > 0:
            # Plot FN curve
            ax.loglog(n_vals, f_vals, 'b-o', linewidth=2, markersize=6, label='FN Curve')

            # Fill area under FN curve
            ax.fill_between(n_vals, f_vals, 1e-12, alpha=0.15, color='blue')

        # Plot criteria
        key = self._criterion_combo.currentData()
        if key and key in FN_CRITERIA:
            crit = FN_CRITERIA[key]
        elif self._criterion:
            crit = self._criterion
        else:
            crit = FN_CRITERIA.get("hse_uk_intolerable")

        if crit:
            n_plot = np.logspace(0, 4, 100)
            f_crit = crit.evaluate_array(n_plot)
            ax.loglog(
                n_plot, f_crit, 'r--', linewidth=2,
                label=f'{crit.name} (intolerable)',
            )

        ax.set_xlabel('Number of Fatalities (N)')
        ax.set_ylabel('Cumulative Frequency F(N) [/yr]')
        ax.set_title('Societal Risk — FN Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.8, max(10, self._fn_data.max_n * 2))

        self._canvas.fig.tight_layout()
        self._canvas.draw()

    def _refresh_summary(self) -> None:
        """Refresh FN summary labels."""
        if self._fn_data is None:
            return

        self._pll_label.setText(f"{self._fn_data.potential_loss_of_life:.6f} /yr")
        self._max_n_label.setText(f"{self._fn_data.max_n}")
        self._total_freq_label.setText(f"{self._fn_data.total_frequency:.4e} /yr")
        self._exp_fat_label.setText(f"{self._fn_data.expected_fatalities:.6f} /yr")

        # ALARP status
        if self._fn_data.alarp_status:
            status_parts = []
            for crit_name, status in self._fn_data.alarp_status.items():
                emoji = {"acceptable": "✅", "alarp": "⚠️", "intolerable": "🚫"}.get(
                    status.value, "❓"
                )
                status_parts.append(f"{emoji} {crit_name}: {status.value}")
            self._alarp_status_label.setText("\n".join(status_parts))
        else:
            self._alarp_status_label.setText("Not evaluated")


# ── Risk Matrix Tab ───────────────────────────────────────────────────

class RiskMatrixTab(QWidget):
    """Risk matrix display with colour-coded grid."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # HTML-based matrix display
        self._matrix_html = QTextEdit()
        self._matrix_html.setReadOnly(True)
        layout.addWidget(self._matrix_html)

        # Quick classification
        class_group = QGroupBox("Quick Risk Classification")
        class_layout = QFormLayout(class_group)

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1e-15, 100.0)
        self._freq_spin.setValue(1e-4)
        self._freq_spin.setDecimals(10)
        self._freq_spin.setSuffix(" /yr")
        class_layout.addRow("Frequency:", self._freq_spin)

        self._fat_spin = QDoubleSpinBox()
        self._fat_spin.setRange(0.0, 10000.0)
        self._fat_spin.setValue(0.1)
        self._fat_spin.setSuffix(" fatalities")
        class_layout.addRow("Expected Fatalities:", self._fat_spin)

        classify_btn = QPushButton("Classify")
        classify_btn.clicked.connect(self._on_classify)
        class_layout.addRow("", classify_btn)

        self._like_label = QLabel("—")
        class_layout.addRow("Likelihood:", self._like_label)

        self._cons_label = QLabel("—")
        class_layout.addRow("Consequence:", self._cons_label)

        self._risk_label = QLabel("—")
        self._risk_label.setFont(QFont("sans-serif", 14, QFont.Weight.Bold))
        class_layout.addRow("Risk Level:", self._risk_label)

        layout.addWidget(class_group)
        self._refresh_matrix()

    def _refresh_matrix(self) -> None:
        """Rebuild the risk matrix HTML display."""
        try:
            html = risk_matrix_html(include_legend=True)
            self._matrix_html.setHtml(html)
        except Exception:
            self._matrix_html.setPlainText("Risk matrix display error")

    def _on_classify(self) -> None:
        """Classify a frequency and consequence pair."""
        freq = self._freq_spin.value()
        fat = self._fat_spin.value()

        like = classify_likelihood(freq)
        cons = classify_consequence(fat)
        rl = risk_level(like, cons)

        self._like_label.setText(f"{like.label} ({like.frequency_range})")
        self._cons_label.setText(f"{cons.label} ({cons.fatality_range})")
        self._risk_label.setText(rl.label)
        self._risk_label.setStyleSheet(
            f"color: {rl.hex_color}; padding: 4px; "
            f"background-color: {self._lighten_color(rl.hex_color, 0.7)}; "
            f"border-radius: 4px;"
        )

    @staticmethod
    def _lighten_color(hex_color: str, factor: float) -> str:
        """Lighten a hex color by factor."""
        c = QColor(hex_color)
        r = min(255, int(c.red() + (255 - c.red()) * factor))
        g = min(255, int(c.green() + (255 - c.green()) * factor))
        b = min(255, int(c.blue() + (255 - c.blue()) * factor))
        return QColor(r, g, b).name()

    def set_scenario_risks(self, scenarios: list) -> None:
        """Classify and display risks for a list of scenarios."""
        # This is a placeholder for future scenario-table display
        pass



# ── Combined Results Widget ───────────────────────────────────────────

class QRAResultsPanel(QWidget):
    """Combined QRA results display with tabs.

    This is the main results widget that should be placed in a
    dock or tab in the main window.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pipeline_data: Optional[dict] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Quantitative Risk Assessment Results")
        title.setFont(QFont("sans-serif", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget for results
        self._tabs = QTabWidget()

        self._ir_tab = IRContourTab()
        self._fn_tab = FNCurveTab()
        self._matrix_tab = RiskMatrixTab()
        self._summary_tab = QRASummaryTab()

        self._tabs.addTab(self._ir_tab, "📍 Individual Risk")
        self._tabs.addTab(self._fn_tab, "📉 FN Curve")
        self._tabs.addTab(self._matrix_tab, "📊 Risk Matrix")
        self._tabs.addTab(self._summary_tab, "📋 Summary")

        layout.addWidget(self._tabs)

    def set_pipeline_result(self, data: dict) -> None:
        """Set results from end-to-end QRAPipeline output.

        Args:
            data: Dict with keys 'lsir_data', 'irpa_data', 'pll_total',
                  'pll_detail', 'fn_data', 'dominant', 'alarp', etc.
        """
        self._pipeline_data = data

        # Update summary tab with pipeline data
        self._summary_tab.set_pipeline_summary(data)

        # Update IR contour tab if LSIR data available
        lsir = data.get("lsir_data", {})
        if lsir and hasattr(self._ir_tab, 'set_lsir_data'):
            self._ir_tab.set_lsir_data(lsir)

        # Update FN curve tab if FN data available
        fn = data.get("fn_data")
        if fn and hasattr(self._fn_tab, 'set_pipeline_fn'):
            self._fn_tab.set_pipeline_fn(fn)

    def set_result(self, data) -> None:
        """Generic set_result — auto-detects legacy or pipeline data format."""
        if isinstance(data, dict):
            if "pipeline_result" in data or "lsir_data" in data or "pll_total" in data:
                # Pipeline format
                self.set_pipeline_result(data)
                return
            # Legacy format
            ir_result = data.get("ir_grid")
            fn_data = data.get("fn_curve")
            if ir_result and hasattr(ir_result, 'x_coords'):
                self._ir_tab.set_result(ir_result)
            elif ir_result and isinstance(ir_result, dict):
                # dict IR grid — cannot contour, but can show thresholds
                if hasattr(self._ir_tab, 'set_lsir_data'):
                    self._ir_tab.set_lsir_data(ir_result)
            if fn_data and hasattr(self._fn_tab, 'set_result'):
                self._fn_tab.set_result(fn_data)
        elif hasattr(data, 'n_values'):
            # FNData object
            self._fn_tab.set_result(data)
        elif hasattr(data, 'ir_grid'):
            # IndividualRiskResult object
            self._ir_tab.set_result(data)


# ── Summary Tab ───────────────────────────────────────────────────────

class QRASummaryTab(QWidget):
    """QRA executive summary tab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setMinimumHeight(200)
        layout.addWidget(self._summary_text)

        # Scenario contributions table
        contrib_group = QGroupBox("Scenario Contributions")
        contrib_layout = QVBoxLayout(contrib_group)

        self._contrib_table = QTableWidget(0, 5)
        self._contrib_table.setHorizontalHeaderLabels([
            "Scenario", "Frequency (/yr)", "Fatalities", "Risk Level", "Contribution %"
        ])
        self._contrib_table.horizontalHeader().setStretchLastSection(True)
        contrib_layout.addWidget(self._contrib_table)

        layout.addWidget(contrib_group)

        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        export_btn = QPushButton("📤 Export QRA Report")
        export_btn.clicked.connect(self._on_export)
        export_layout.addWidget(export_btn)
        layout.addLayout(export_layout)

    def set_summary(
        self,
        ir_result: Optional[IndividualRiskResult] = None,
        fn_data: Optional[FNData] = None,
        scenarios: Optional[list] = None,
    ) -> None:
        """Populate the summary with QRA results."""
        lines = []
        lines.append("=" * 60)
        lines.append("QRA EXECUTIVE SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        if ir_result:
            lines.append("── INDIVIDUAL RISK ──")
            lines.append(f"  Maximum IR:          {ir_result.max_ir:.4e} /yr")
            ir_source = get_ir_at_source(ir_result)
            lines.append(f"  IR at Source:        {ir_source:.4e} /yr")
            ir_100 = get_ir_at_distance(ir_result, 100.0)
            lines.append(f"  IR at 100 m:         {ir_100:.4e} /yr")
            ir_500 = get_ir_at_distance(ir_result, 500.0)
            lines.append(f"  IR at 500 m:         {ir_500:.4e} /yr")
            lines.append("")

        if fn_data:
            lines.append("── SOCIETAL RISK ──")
            lines.append(f"  PLL:                 {fn_data.potential_loss_of_life:.6f} /yr")
            lines.append(f"  Max Fatalities:      {fn_data.max_n}")
            lines.append(f"  Total Frequency:     {fn_data.total_frequency:.4e} /yr")
            lines.append(f"  Expected Fatalities: {fn_data.expected_fatalities:.6f} /yr")
            lines.append("")
            lines.append("  ALARP Status:")
            for crit_name, status in fn_data.alarp_status.items():
                lines.append(f"    {crit_name}: {status.value}")
            lines.append("")

        if scenarios:
            lines.append("── SCENARIOS ──")
            total_prob = sum(getattr(s, 'probability', 0) for s in scenarios)
            for s in scenarios:
                prob = getattr(s, 'probability', 0)
                pct = (prob / total_prob * 100) if total_prob > 0 else 0
                ct = getattr(s, 'consequence_type', 'unknown')
                lines.append(f"  • {s.name}: {prob:.4e}/yr ({pct:.1f}%) — {ct}")
            lines.append("")

        lines.append("=" * 60)
        self._summary_text.setPlainText("\n".join(lines))

        # Scenario contributions table
        if scenarios:
            self._contrib_table.setRowCount(len(scenarios))
            total_prob = sum(getattr(s, 'probability', 0) for s in scenarios)
            for i, s in enumerate(scenarios):
                prob = getattr(s, 'probability', 0)
                self._contrib_table.setItem(i, 0, QTableWidgetItem(getattr(s, 'name', 'Unknown')))

                freq_item = QTableWidgetItem(f"{prob:.4e}")
                freq_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._contrib_table.setItem(i, 1, freq_item)

                params = getattr(s, 'consequence_params', {})
                n_fat = params.get('expected_fatalities', 0.0)
                self._contrib_table.setItem(i, 2, QTableWidgetItem(f"{n_fat:.2f}"))

                rl = risk_level_from_values(prob, n_fat)
                rl_item = QTableWidgetItem(rl.label)
                rl_item.setBackground(QColor(rl.hex_color))
                self._contrib_table.setItem(i, 3, rl_item)

                pct = (prob / total_prob * 100) if total_prob > 0 else 0
                pct_item = QTableWidgetItem(f"{pct:.1f}%")
                pct_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._contrib_table.setItem(i, 4, pct_item)

            self._contrib_table.resizeColumnsToContents()

    def set_pipeline_summary(self, data: dict) -> None:
        """Populate summary from end-to-end QRAPipeline results."""
        lines = []
        lines.append("=" * 60)
        lines.append("QRA PIPELINE SUMMARY (End-to-End)")
        lines.append("=" * 60)
        lines.append("")

        n_scenarios = data.get("scenario_count", 0)
        lines.append(f"  Scenarios evaluated:  {n_scenarios}")
        lines.append("")

        # LSIR
        lsir = data.get("lsir_data", {})
        if lsir:
            lines.append("── LOCATION-SPECIFIC INDIVIDUAL RISK (LSIR) ──")
            for loc, val in sorted(lsir.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {loc:>20s}:  {val:.4e} /yr")
            if len(lsir) > 10:
                lines.append(f"  ... ({len(lsir)} total locations)")
            lines.append("")

        # IRPA
        irpa = data.get("irpa_data", {})
        if irpa:
            lines.append("── INDIVIDUAL RISK PER ANNUM (IRPA) ──")
            for wg, val in irpa.items():
                lines.append(f"  {wg:>20s}:  {val:.4e} /yr")
            lines.append("")

        # PLL
        pll_total = data.get("pll_total", 0.0)
        lines.append("── POTENTIAL LOSS OF LIFE (PLL) ──")
        lines.append(f"  Total PLL:            {pll_total:.6f} /yr")
        lines.append("")

        # ALARP
        alarp = data.get("alarp", {})
        if alarp:
            lines.append("── ALARP ASSESSMENT (HSE UK Criteria) ──")
            for wg, status in alarp.items():
                irpa_val = irpa.get(wg, 0) if irpa else 0
                emoji = "🚫" if "INTOLERABLE" in status else (
                        "⚠️" if "ALARP" in status else "✅")
                lines.append(f"  {emoji} {wg:>20s}:  {status}")
            lines.append("")

        # Dominant contributors
        dominant = data.get("dominant", [])
        if dominant:
            lines.append("── DOMINANT RISK CONTRIBUTORS ──")
            for d in dominant[:5]:
                if isinstance(d, dict):
                    lines.append(
                        f"  • {d.get('scenario', '?'):40s}  "
                        f"{d.get('frequency', 0):.4e} /yr"
                    )
            lines.append("")

        # Warnings
        warnings = data.get("warnings", [])
        if warnings:
            lines.append("── WARNINGS ──")
            for w in warnings[:5]:
                lines.append(f"  ⚠️ {w}")
            lines.append("")

        lines.append("=" * 60)
        self._summary_text.setPlainText("\n".join(lines))

        # Update contributions table from dominant scenarios
        if dominant:
            self._contrib_table.setRowCount(len(dominant))
            for i, d in enumerate(dominant):
                if isinstance(d, dict):
                    self._contrib_table.setItem(i, 0,
                        QTableWidgetItem(d.get("scenario", "?")))
                    freq = d.get("frequency", 0)
                    freq_item = QTableWidgetItem(f"{freq:.4e}")
                    freq_item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self._contrib_table.setItem(i, 1, freq_item)
                    self._contrib_table.setItem(i, 2,
                        QTableWidgetItem(d.get("outcome", "?")))
                    rl = risk_level_from_values(freq, 1)
                    rl_item = QTableWidgetItem(rl.label)
                    rl_item.setBackground(QColor(rl.hex_color))
                    self._contrib_table.setItem(i, 3, rl_item)
                    pct_item = QTableWidgetItem("—")
                    self._contrib_table.setItem(i, 4, pct_item)
            self._contrib_table.resizeColumnsToContents()

    def _on_export(self) -> None:
        """Export QRA summary to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export QRA Report", "", "Text Files (*.txt);;HTML Files (*.html)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    f.write(self._summary_text.toPlainText())
                QMessageBox.information(self, "Exported", f"Report saved to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))


# ── Convenience imports at module level ───────────────────────────────

# Placeholder for risk_level_from_values (from risk_matrix)
def risk_level_from_values(frequency: float, fatalities: float) -> RiskLevel:
    """Convenience wrapper."""
    return risk_level(
        classify_likelihood(frequency),
        classify_consequence(fatalities),
    )


class QRAResultsDock(QWidget):
    """Dock-wrapped QRA results panel (for main window integration)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._panel = QRAResultsPanel()
        layout.addWidget(self._panel)

    def panel(self) -> QRAResultsPanel:
        return self._panel
