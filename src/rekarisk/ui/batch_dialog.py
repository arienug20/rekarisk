"""
Rekarisk UI — Batch Runner Dialog.

PyQt6 dialog for configuring and running batch consequence scenarios.
Features:
  - Scenario list with add/remove from templates
  - Weather set and substance set selectors
  - Combination mode toggle
  - Progress bar with live status
  - Results summary table
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QSplitter, QComboBox,
    QHeaderView, QMessageBox, QWidget, QDialogButtonBox,
    QApplication,
)

from ..analysis.batch_runner import BatchInput, BatchResult, BatchRunner


class _BatchWorker(QThread):
    """Worker thread for batch execution."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, batch_input: BatchInput):
        super().__init__()
        self.batch_input = batch_input
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            runner = BatchRunner()
            result = runner.run(self.batch_input, callback=self._on_progress)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_progress(self, n_done: int, n_total: int, status: str):
        if self._cancelled:
            raise KeyboardInterrupt("Batch cancelled by user")
        self.progress.emit(n_done, n_total, status)


class BatchDialog(QDialog):
    """Dialog for configuring and running batch consequence scenarios.

    Usage:
        dialog = BatchDialog(
            model_function=my_dispersion_model,
            scenario_templates={"Light Release": {...}, "Heavy Release": {...}},
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.batch_result
            print(result.summary_table)
    """

    def __init__(
        self,
        model_function: Callable,
        scenario_templates: Optional[Dict[str, Dict[str, Any]]] = None,
        weather_options: Optional[List[Dict[str, Any]]] = None,
        substance_options: Optional[List[Any]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Runner")
        self.setMinimumSize(800, 600)
        self.resize(950, 700)

        self.model_function = model_function
        self.scenario_templates = scenario_templates or {}
        self.weather_options = weather_options or []
        self.substance_options = substance_options or []

        self._scenarios: List[Dict[str, Any]] = []
        self._selected_weather: List[Dict[str, Any]] = []
        self._selected_substances: List[Any] = []
        self._worker: Optional[_BatchWorker] = None
        self.batch_result: Optional[BatchResult] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- Top: Scenario Management ----
        scenario_group = QGroupBox("Scenarios")
        scenario_layout = QVBoxLayout(scenario_group)

        # Add/remove buttons + template dropdown
        btn_layout = QHBoxLayout()

        self._template_combo = QComboBox()
        self._template_combo.setToolTip("Select a scenario template")
        self._template_combo.addItem("-- Templates --")
        for name in self.scenario_templates:
            self._template_combo.addItem(name)

        self._add_btn = QPushButton("+ Add")
        self._add_btn.setToolTip("Add a scenario from template")
        self._remove_btn = QPushButton("− Remove")
        self._remove_btn.setToolTip("Remove selected scenario")

        btn_layout.addWidget(QLabel("Template:"))
        btn_layout.addWidget(self._template_combo, 1)
        btn_layout.addWidget(self._add_btn)
        btn_layout.addWidget(self._remove_btn)
        scenario_layout.addLayout(btn_layout)

        self._scenario_table = QTableWidget(0, 2)
        self._scenario_table.setHorizontalHeaderLabels(["#", "Scenario Description"])
        self._scenario_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._scenario_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._scenario_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        scenario_layout.addWidget(self._scenario_table)
        layout.addWidget(scenario_group)

        # ---- Middle: Options ----
        options_group = QGroupBox("Run Options")
        options_layout = QHBoxLayout(options_group)

        self._combo_check = QCheckBox("Run all combinations")
        self._combo_check.setToolTip(
            "If checked, run every weather × substance × scenario combination"
        )

        self._parallel_spin = QSpinBox()
        self._parallel_spin.setRange(0, 8)
        self._parallel_spin.setValue(0)
        self._parallel_spin.setToolTip("Number of parallel workers (0 = sequential)")

        options_layout.addWidget(self._combo_check)
        options_layout.addStretch()
        options_layout.addWidget(QLabel("Parallel workers:"))
        options_layout.addWidget(self._parallel_spin)

        layout.addWidget(options_group)

        # ---- Middle: Run Controls ----
        run_layout = QHBoxLayout()

        self._run_btn = QPushButton("▶ Run Batch")
        self._run_btn.setToolTip("Execute all scenarios in sequence")
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 20px; }"
        )

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)

        run_layout.addWidget(self._run_btn)
        run_layout.addWidget(self._cancel_btn)
        run_layout.addWidget(self._progress_bar, 1)
        layout.addLayout(run_layout)

        # Status label
        self._status_label = QLabel("Ready. Add scenarios to begin.")
        layout.addWidget(self._status_label)

        # ---- Results Area ----
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Summary table
        self._results_table = QTableWidget(0, 3)
        self._results_table.setHorizontalHeaderLabels([
            "Scenario", "Status", "Key Metric"
        ])
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        splitter.addWidget(self._results_table)

        # Log output
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumHeight(150)
        splitter.addWidget(self._log_area)

        results_layout.addWidget(splitter)
        layout.addWidget(results_group)

        # ---- Bottom: Close button ----
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _connect_signals(self):
        self._add_btn.clicked.connect(self._add_scenario)
        self._remove_btn.clicked.connect(self._remove_scenario)
        self._run_btn.clicked.connect(self._run_batch)
        self._cancel_btn.clicked.connect(self._cancel_batch)

    def _add_scenario(self):
        template_name = self._template_combo.currentText()
        if template_name == "-- Templates --" or template_name not in self.scenario_templates:
            QMessageBox.warning(self, "No Template", "Select a template first.")
            return

        scenario = dict(self.scenario_templates[template_name])
        self._scenarios.append(scenario)

        row = self._scenario_table.rowCount()
        self._scenario_table.insertRow(row)
        self._scenario_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        desc = self._describe_scenario(scenario, template_name)
        self._scenario_table.setItem(row, 1, QTableWidgetItem(desc))

        self._update_status()

    def _remove_scenario(self):
        selected = self._scenario_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self._scenarios):
            self._scenarios.pop(row)
            self._scenario_table.removeRow(row)
            # Re-number
            for i in range(self._scenario_table.rowCount()):
                self._scenario_table.item(i, 0).setText(str(i + 1))
        self._update_status()

    def _describe_scenario(self, scenario: Dict[str, Any], template: str) -> str:
        """Create a human-readable scenario description."""
        parts = [f"[{template}]"]
        for key in ["source_rate", "mass_rate", "wind_speed",
                     "stability_class", "release_height",
                     "mass", "duration"]:
            if key in scenario:
                val = scenario[key]
                parts.append(f"{key}={val}")
        return ", ".join(parts)

    def _update_status(self):
        n = len(self._scenarios)
        if n == 0:
            self._status_label.setText("Ready. Add scenarios to begin.")
            self._run_btn.setEnabled(False)
        else:
            self._status_label.setText(f"{n} scenario(s) queued. Click Run Batch.")
            self._run_btn.setEnabled(True)

    def _run_batch(self):
        if not self._scenarios:
            QMessageBox.warning(self, "No Scenarios", "Add at least one scenario first.")
            return

        # Build BatchInput
        batch_input = BatchInput(
            scenarios=list(self._scenarios),
            weather_set=self._selected_weather,
            substance_list=self._selected_substances,
            combinations=self._combo_check.isChecked(),
            model_function=self.model_function,
            parallel=self._parallel_spin.value(),
            verbose=False,
        )

        # Disable UI
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Running batch...")
        self._results_table.setRowCount(0)
        self._log_area.clear()

        # Start worker
        self._worker = _BatchWorker(batch_input)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel_batch(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.terminate()
            self._worker.wait(2000)
            self._status_label.setText("Batch cancelled.")
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            self._log_area.append("[CANCELLED] Batch execution stopped by user.")

    def _on_progress(self, n_done: int, n_total: int, status: str):
        pct = int(n_done / max(1, n_total) * 100)
        self._progress_bar.setValue(pct)
        self._progress_bar.setFormat(f"{n_done}/{n_total}")
        self._status_label.setText(f"Running: {status}")
        self._log_area.append(f"[{n_done}/{n_total}] {status}")

    def _on_finished(self, result: BatchResult):
        self.batch_result = result
        self._progress_bar.setValue(100)
        self._progress_bar.setFormat(f"{result.success_count}/{result.total_scenarios}")

        n_ok = result.success_count
        n_fail = result.failure_count
        t_elapsed = result.elapsed_time

        self._status_label.setText(
            f"Done. {n_ok} succeeded, {n_fail} failed in {t_elapsed:.1f}s."
        )

        # Populate results table
        self._results_table.setRowCount(0)
        for row in result.summary_table:
            r = self._results_table.rowCount()
            self._results_table.insertRow(r)

            sid = row.get("scenario_id", "?")
            self._results_table.setItem(r, 0, QTableWidgetItem(str(sid)))

            if "error" in row:
                self._results_table.setItem(r, 1, QTableWidgetItem("FAILED"))
                self._results_table.setItem(r, 2, QTableWidgetItem(row["error"]))
            else:
                self._results_table.setItem(r, 1, QTableWidgetItem("OK"))
                # Show first available metric
                for k in ["max_concentration", "max_flux", "heat_flux",
                           "overpressure", "impact_distance", "probit_value"]:
                    if k in row:
                        self._results_table.setItem(
                            r, 2,
                            QTableWidgetItem(f"{k}: {row[k]:.4g}")
                        )
                        break
                else:
                    self._results_table.setItem(r, 2, QTableWidgetItem("—"))

        # Log statistics
        if result.statistics:
            self._log_area.append("\n--- Aggregate Statistics ---")
            for metric, stats_dict in result.statistics.items():
                self._log_area.append(
                    f"  {metric}: "
                    f"μ={stats_dict.get('mean', 0):.4g}, "
                    f"σ={stats_dict.get('std', 0):.4g}, "
                    f"[{stats_dict.get('min', 0):.4g}, {stats_dict.get('max', 0):.4g}]"
                )

        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        self._log_area.append(f"[ERROR] {msg}")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def set_weather_options(self, options: List[Dict[str, Any]]):
        """Set available weather conditions."""
        self.weather_options = options

    def set_substance_options(self, options: List[Any]):
        """Set available substances."""
        self.substance_options = options

    def results_summary(self) -> str:
        """Return a text summary of the batch results."""
        if self.batch_result is None:
            return "No batch results available."

        res = self.batch_result
        lines = [
            f"Batch Results: {res.success_count}/{res.total_scenarios} succeeded "
            f"({res.failure_count} failed) in {res.elapsed_time:.1f}s",
            "",
        ]
        if res.statistics:
            lines.append("Aggregate Statistics:")
            for metric, stats_dict in res.statistics.items():
                lines.append(
                    f"  {metric}: "
                    f"mean={stats_dict['mean']:.4g}, "
                    f"std={stats_dict['std']:.4g}, "
                    f"range=[{stats_dict['min']:.4g}, {stats_dict['max']:.4g}]"
                )
        return "\n".join(lines)
