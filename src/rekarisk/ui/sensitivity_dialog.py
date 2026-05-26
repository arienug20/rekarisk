"""
Rekarisk UI — Sensitivity Analysis Dialog.

PyQt6 dialog for One-At-a-Time (OAT) parameter sensitivity analysis.
Features:
  - Parameter table: name, base value, min, max (editable)
  - Output metric selector
  - Run button with results table showing ranked parameters
  - Tornado chart preview (matplotlib embedded)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QSpinBox, QComboBox,
    QProgressBar, QGroupBox, QSplitter, QHeaderView,
    QMessageBox, QWidget, QDialogButtonBox,
    QApplication,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..analysis.sensitivity import (
    SensitivityInput, SensitivityResult, run_oat, tornado_data, parameter_ranks,
)


class _SensitivityWorker(QThread):
    """Worker thread for OAT sensitivity analysis."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, sensitivity_input: SensitivityInput):
        super().__init__()
        self.sensitivity_input = sensitivity_input

    def run(self):
        try:
            result = run_oat(self.sensitivity_input)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SensitivityDialog(QDialog):
    """Dialog for parameter sensitivity analysis.

    Usage:
        dialog = SensitivityDialog(
            model_function=my_plume_model,
            base_params={"source_rate": 5.0, "wind_speed": 3.0, "release_height": 10.0},
            output_key="max_concentration",
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.sensitivity_result
            print(result.rankings)
    """

    def __init__(
        self,
        model_function: Callable,
        base_params: Optional[Dict[str, Any]] = None,
        output_key: Optional[str] = None,
        output_keys: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Sensitivity Analysis")
        self.setMinimumSize(850, 600)
        self.resize(900, 700)

        self.model_function = model_function
        self.base_params = base_params or {}
        self.output_key = output_key
        self.output_keys = output_keys or []

        self._worker: Optional[_SensitivityWorker] = None
        self.sensitivity_result: Optional[SensitivityResult] = None

        self._setup_ui()
        self._populate_params()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- Top: Parameter Table ----
        param_group = QGroupBox("Parameters (base value and OAT range)")
        param_layout = QVBoxLayout(param_group)

        self._param_table = QTableWidget(0, 5)
        self._param_table.setHorizontalHeaderLabels([
            "Parameter", "Base Value", "Min (−20%)", "Max (+20%)", "Include"
        ])
        self._param_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )

        param_layout.addWidget(self._param_table)

        # Auto-fill button
        btn_layout = QHBoxLayout()
        self._auto_range_btn = QPushButton("Auto-fill ±20% Ranges")
        self._auto_range_btn.setToolTip(
            "Set min/max to ±20% of base value for all parameters"
        )
        self._select_all_btn = QPushButton("Select All")
        self._deselect_all_btn = QPushButton("Deselect All")
        btn_layout.addWidget(self._auto_range_btn)
        btn_layout.addWidget(self._select_all_btn)
        btn_layout.addWidget(self._deselect_all_btn)
        btn_layout.addStretch()
        param_layout.addLayout(btn_layout)

        layout.addWidget(param_group)

        # ---- Output selector ----
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Metric:"))
        self._output_combo = QComboBox()
        self._output_combo.setEditable(True)
        self._output_combo.setToolTip(
            "Metric to analyze (e.g., max_concentration, heat_flux)"
        )
        # Populate with common metrics
        default_metrics = [
            "max_concentration", "max_flux", "heat_flux",
            "overpressure", "impact_distance", "max_distance",
            "probit_value", "probability_of_death",
        ]
        for m in default_metrics:
            self._output_combo.addItem(m)
        if self.output_keys:
            for k in self.output_keys:
                if k not in default_metrics:
                    self._output_combo.addItem(k)

        output_layout.addWidget(self._output_combo, 1)
        output_layout.addStretch()
        layout.addLayout(output_layout)

        # ---- Run Controls ----
        run_layout = QHBoxLayout()

        self._run_btn = QPushButton("▶ Run Sensitivity")
        self._run_btn.setToolTip("Run OAT sensitivity analysis")
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 20px; }"
        )
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

        run_layout.addWidget(self._run_btn)
        run_layout.addWidget(self._cancel_btn)
        run_layout.addWidget(self._progress_bar, 1)
        layout.addLayout(run_layout)

        self._status_label = QLabel("Ready. Configure parameters and click Run.")
        layout.addWidget(self._status_label)

        # ---- Results Area ----
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Results table
        self._results_table = QTableWidget(0, 6)
        self._results_table.setHorizontalHeaderLabels([
            "Rank", "Parameter", "Low Output", "High Output",
            "Range", "Sensitivity %"
        ])
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        splitter.addWidget(self._results_table)

        # Tornado chart
        self._tornado_canvas = FigureCanvasQTAgg(Figure(figsize=(8, 3)))
        self._tornado_ax = self._tornado_canvas.figure.subplots()
        splitter.addWidget(self._tornado_canvas)

        layout.addWidget(splitter)

        # ---- Bottom ----
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Connect signals
        self._run_btn.clicked.connect(self._run_analysis)
        self._cancel_btn.clicked.connect(self._cancel_analysis)
        self._auto_range_btn.clicked.connect(self._auto_fill_ranges)
        self._select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self._deselect_all_btn.clicked.connect(lambda: self._set_all_checked(False))

    def _populate_params(self):
        """Populate parameter table from base_params."""
        self._param_table.setRowCount(0)
        for name, value in self.base_params.items():
            if not isinstance(value, (int, float, np.floating)) or isinstance(value, bool):
                continue

            bv = float(value)
            row = self._param_table.rowCount()
            self._param_table.insertRow(row)

            # Parameter name
            name_item = QTableWidgetItem(str(name))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(row, 0, name_item)

            # Base value
            base_item = QTableWidgetItem(f"{bv:.6g}")
            self._param_table.setItem(row, 1, base_item)

            # Min
            min_item = QTableWidgetItem(f"{bv * 0.8:.4g}")
            self._param_table.setItem(row, 2, min_item)

            # Max
            max_item = QTableWidgetItem(f"{bv * 1.2:.4g}")
            self._param_table.setItem(row, 3, max_item)

            # Include checkbox
            from PyQt6.QtWidgets import QCheckBox
            cb = QCheckBox()
            cb.setChecked(True)
            self._param_table.setCellWidget(row, 4, cb)

    def _get_selected_params(self) -> Dict[str, Any]:
        """Get the parameters and ranges as configured in the table."""
        params: Dict[str, Any] = {}
        base_case: Dict[str, Any] = {}
        param_ranges: Dict[str, Tuple[float, float]] = {}

        for row in range(self._param_table.rowCount()):
            name = self._param_table.item(row, 0)
            base = self._param_table.item(row, 1)
            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            cb = self._param_table.cellWidget(row, 4)

            if name is None or base is None:
                continue
            if cb and not cb.isChecked():
                continue

            try:
                name_str = str(name.text()).strip()
                base_val = float(base.text().strip())
                min_val = float(min_item.text().strip()) if min_item else base_val * 0.8
                max_val = float(max_item.text().strip()) if max_item else base_val * 1.2

                if min_val > max_val:
                    min_val, max_val = max_val, min_val

                base_case[name_str] = base_val
                param_ranges[name_str] = (min_val, max_val)
            except (ValueError, TypeError):
                continue

        return {
            "base_case": base_case,
            "parameters": param_ranges,
        }

    def _auto_fill_ranges(self):
        """Set min/max to ±20% of base for all rows."""
        for row in range(self._param_table.rowCount()):
            base_item = self._param_table.item(row, 1)
            if base_item is None:
                continue
            try:
                bv = float(base_item.text().strip())
                if bv != 0:
                    min_val = bv * 0.8
                    max_val = bv * 1.2
                else:
                    min_val = -1.0
                    max_val = 1.0
            except ValueError:
                continue

            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            if min_item:
                min_item.setText(f"{min_val:.4g}")
            if max_item:
                max_item.setText(f"{max_val:.4g}")

    def _set_all_checked(self, checked: bool):
        """Check/uncheck all Include checkboxes."""
        for row in range(self._param_table.rowCount()):
            cb = self._param_table.cellWidget(row, 4)
            if cb:
                cb.setChecked(checked)

    def _run_analysis(self):
        """Collect input and run sensitivity analysis."""
        params = self._get_selected_params()
        if not params["parameters"]:
            QMessageBox.warning(self, "No Parameters",
                                "No parameters selected for analysis.")
            return

        # Determine output key
        output_key = self._output_combo.currentText().strip()
        if not output_key:
            output_key = None
        elif output_key not in _VALID_METRICS:
            output_key = None

        si = SensitivityInput(
            base_case=params["base_case"],
            parameters=params["parameters"],
            model_function=self.model_function,
            output_key=output_key,
        )

        # Disable UI
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Running sensitivity analysis...")
        self._results_table.setRowCount(0)

        self._worker = _SensitivityWorker(si)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel_analysis(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
            self._status_label.setText("Analysis cancelled.")
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, n_done: int, n_total: int, status: str):
        pct = int(n_done / max(1, n_total) * 100)
        self._progress_bar.setValue(pct)
        self._status_label.setText(f"Evaluating: {status}")

    def _on_finished(self, result: SensitivityResult):
        self.sensitivity_result = result
        self._progress_bar.setValue(100)

        if result.model_errors:
            self._status_label.setText(
                f"Done. {result.success_count} params evaluated, "
                f"{result.error_count} errors."
            )
        else:
            self._status_label.setText(
                f"Done. {result.success_count} parameters ranked."
            )

        # Populate results table
        ranks = parameter_ranks(result)
        self._results_table.setRowCount(0)
        for r in ranks:
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            self._results_table.setItem(row, 0, QTableWidgetItem(str(r["rank"])))
            self._results_table.setItem(row, 1, QTableWidgetItem(str(r["parameter"])))
            self._results_table.setItem(row, 2, QTableWidgetItem(f"{r['low_output']:.4g}"))
            self._results_table.setItem(row, 3, QTableWidgetItem(f"{r['high_output']:.4g}"))
            self._results_table.setItem(row, 4, QTableWidgetItem(f"{r['range']:.4g}"))
            self._results_table.setItem(row, 5, QTableWidgetItem(f"{r['range_pct']:.1f}%"))

        # Draw tornado chart
        self._draw_tornado(result)

        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _draw_tornado(self, result: SensitivityResult):
        """Draw or update the tornado chart."""
        self._tornado_ax.clear()

        labels, base, low_vals, high_vals = tornado_data(result, top_n=15)
        if not labels:
            self._tornado_ax.text(0.5, 0.5, "No data",
                                   ha="center", va="center",
                                   transform=self._tornado_ax.transAxes)
            self._tornado_canvas.draw()
            return

        n = len(labels)
        y_pos = range(n)

        # Compute bar positions relative to base
        bar_low = [base - v for v in low_vals]
        bar_high = [v - base for v in high_vals]

        colors_low = ["#D32F2F" if b > 0 else "#1976D2" for b in bar_low]
        colors_high = ["#D32F2F" if b > 0 else "#1976D2" for b in bar_high]

        self._tornado_ax.barh(y_pos, bar_low, height=0.6,
                               color=colors_low, alpha=0.85,
                               label="Low (min param)")
        self._tornado_ax.barh(y_pos, bar_high, left=base, height=0.6,
                               color=colors_high, alpha=0.85,
                               label="High (max param)")

        self._tornado_ax.set_yticks(y_pos)
        self._tornado_ax.set_yticklabels(labels, fontsize=9)
        self._tornado_ax.axvline(x=base, color="black", linewidth=1.5,
                                  linestyle="--")
        self._tornado_ax.set_xlabel(f"Output (base = {base:.4g})")
        self._tornado_ax.legend(loc="lower right", fontsize=8)

        self._tornado_canvas.figure.tight_layout()
        self._tornado_canvas.draw()

    def set_model_output_metrics(self, metrics: List[str]):
        """Set available output metric options."""
        self.output_keys = metrics
        current = self._output_combo.currentText()
        self._output_combo.clear()
        for m in metrics:
            self._output_combo.addItem(m)
        if current:
            idx = self._output_combo.findText(current)
            if idx >= 0:
                self._output_combo.setCurrentIndex(idx)


_VALID_METRICS = {
    "max_concentration", "max_flux", "heat_flux", "overpressure",
    "impact_distance", "max_distance", "probit_value",
    "probability_of_death", "risk_individual",
    "pool_radius", "flame_length", "flame_height",
    "surface_emissive_power", "impulse",
}
