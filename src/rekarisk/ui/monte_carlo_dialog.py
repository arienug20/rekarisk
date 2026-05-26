"""
Rekarisk UI — Monte Carlo Simulation Dialog.

PyQt6 dialog for Monte Carlo uncertainty propagation.
Features:
  - Parameter table: name, distribution type, parameters (editable)
  - N samples and confidence level inputs
  - Run button with progress indicator
  - Results: histogram of outputs, statistics table, CI display
  - Correlation heatmap
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QSpinBox, QComboBox,
    QDoubleSpinBox, QProgressBar, QGroupBox, QSplitter,
    QHeaderView, QMessageBox, QWidget, QDialogButtonBox,
    QTabWidget, QTextEdit, QApplication,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..analysis.monte_carlo import (
    Distribution, Normal, LogNormal, Uniform, Triangular, Beta,
    MCInput, MCResult, run_monte_carlo, convergence_check,
    estimate_required_samples, make_distribution,
)


DIST_TYPES = ["Normal", "LogNormal", "Uniform", "Triangular", "Beta"]
DIST_PARAMS_META = {
    "Normal": [("μ  (mean)", 0.0), ("σ  (std)", 1.0)],
    "LogNormal": [("μ  (of ln)", 0.0), ("σ  (of ln)", 1.0)],
    "Uniform": [("a  (min)", 0.0), ("b  (max)", 1.0)],
    "Triangular": [("a  (min)", 0.0), ("mode", 0.5), ("b  (max)", 1.0)],
    "Beta": [("α  (alpha)", 2.0), ("β  (beta)", 2.0)],
}


class _MCWorker(QThread):
    """Worker thread for Monte Carlo simulation."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, mc_input: MCInput):
        super().__init__()
        self.mc_input = mc_input
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = run_monte_carlo(self.mc_input)
            if self._cancelled:
                return
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_progress(self, n_done: int, n_total: int, status: str):
        if self._cancelled:
            raise KeyboardInterrupt("MC cancelled by user")
        self.progress.emit(n_done, n_total, status)


class MonteCarloDialog(QDialog):
    """Dialog for Monte Carlo uncertainty propagation.

    Usage:
        params = {"source_rate": Normal(5.0, 1.0), "wind_speed": Uniform(2.0, 6.0)}
        dialog = MonteCarloDialog(
            model_function=my_plume_model,
            parameters=params,
            output_keys=["max_concentration"],
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.mc_result
            print(result.statistics)
    """

    def __init__(
        self,
        model_function: Callable,
        parameters: Optional[Dict[str, Distribution]] = None,
        output_keys: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Monte Carlo Simulation")
        self.setMinimumSize(900, 650)
        self.resize(950, 750)

        self.model_function = model_function
        self._input_params = parameters or {}
        self._output_keys = output_keys or []

        self._worker: Optional[_MCWorker] = None
        self.mc_result: Optional[MCResult] = None

        self._setup_ui()
        if self._input_params:
            self._populate_params()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- Top: Parameter Table ----
        param_group = QGroupBox("Input Parameters & Distributions")
        param_layout = QVBoxLayout(param_group)

        self._param_table = QTableWidget(0, 5)
        self._param_table.setHorizontalHeaderLabels([
            "Parameter", "Distribution", "Param 1", "Param 2", "Param 3"
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

        # Add parameter button
        btn_layout = QHBoxLayout()
        self._add_param_btn = QPushButton("+ Add Row")
        self._remove_param_btn = QPushButton("− Remove Row")
        btn_layout.addWidget(self._add_param_btn)
        btn_layout.addWidget(self._remove_param_btn)
        btn_layout.addStretch()
        param_layout.addLayout(btn_layout)

        layout.addWidget(param_group)

        # ---- Options ----
        opts_group = QGroupBox("Simulation Options")
        opts_layout = QHBoxLayout(opts_group)

        opts_layout.addWidget(QLabel("N samples:"))
        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(10, 1000000)
        self._n_samples_spin.setValue(1000)
        self._n_samples_spin.setSingleStep(100)
        self._n_samples_spin.setToolTip("Number of Monte Carlo samples")
        opts_layout.addWidget(self._n_samples_spin)

        opts_layout.addSpacing(20)

        opts_layout.addWidget(QLabel("Confidence level:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.50, 0.999)
        self._conf_spin.setValue(0.95)
        self._conf_spin.setSingleStep(0.01)
        self._conf_spin.setDecimals(3)
        self._conf_spin.setToolTip("Confidence level for intervals (e.g., 0.95 = 95%)")
        opts_layout.addWidget(self._conf_spin)

        opts_layout.addSpacing(20)

        opts_layout.addWidget(QLabel("Random seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setSpecialValueText("Random")
        self._seed_spin.setToolTip("Seed for reproducibility (-1 = random)")
        opts_layout.addWidget(self._seed_spin)

        opts_layout.addStretch()
        layout.addWidget(opts_group)

        # ---- Run Controls ----
        run_layout = QHBoxLayout()

        self._run_btn = QPushButton("▶ Run Simulation")
        self._run_btn.setToolTip("Run Monte Carlo simulation")
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

        # ---- Results Tabs ----
        self._results_tabs = QTabWidget()

        # Tab 1: Statistics
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self._stats_table = QTableWidget(0, 7)
        self._stats_table.setHorizontalHeaderLabels([
            "Metric", "Mean", "Std", "Median", "P5", "P95", "CI"
        ])
        self._stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        stats_layout.addWidget(self._stats_table)
        self._results_tabs.addTab(stats_widget, "Statistics")

        # Tab 2: Histogram
        self._hist_canvas = FigureCanvasQTAgg(Figure(figsize=(8, 4)))
        self._results_tabs.addTab(self._hist_canvas, "Histogram")

        # Tab 3: Convergence
        conv_widget = QWidget()
        conv_layout = QVBoxLayout(conv_widget)
        self._conv_text = QTextEdit()
        self._conv_text.setReadOnly(True)
        conv_layout.addWidget(self._conv_text)
        self._results_tabs.addTab(conv_widget, "Convergence")

        layout.addWidget(self._results_tabs)

        # ---- Bottom ----
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Connect signals
        self._run_btn.clicked.connect(self._run_simulation)
        self._cancel_btn.clicked.connect(self._cancel_simulation)
        self._add_param_btn.clicked.connect(self._add_param_row)
        self._remove_param_btn.clicked.connect(self._remove_param_row)

    def _populate_params(self):
        """Populate parameter table from _input_params dict."""
        self._param_table.setRowCount(0)
        for name, dist in self._input_params.items():
            self._add_row_from_dist(name, dist)

    def _add_row_from_dist(self, name: str, dist: Distribution):
        """Add a row from an existing Distribution."""
        row = self._param_table.rowCount()
        self._param_table.insertRow(row)

        # Param name
        name_item = QTableWidgetItem(str(name))
        self._param_table.setItem(row, 0, name_item)

        # Distribution type
        dist_combo = QComboBox()
        dist_combo.addItems(DIST_TYPES)
        type_name = type(dist).__name__
        if type_name in DIST_TYPES:
            dist_combo.setCurrentText(type_name)
        self._param_table.setCellWidget(row, 1, dist_combo)

        # Extract params based on type
        if isinstance(dist, Normal):
            params = [dist.mu, dist.sigma, ""]
        elif isinstance(dist, LogNormal):
            params = [dist.mu, dist.sigma, ""]
        elif isinstance(dist, Uniform):
            params = [dist.a, dist.b, ""]
        elif isinstance(dist, Triangular):
            params = [dist.a, dist.mode_value, dist.b]
        elif isinstance(dist, Beta):
            params = [dist.alpha, dist.beta_param, ""]
        else:
            params = ["", "", ""]

        for ci, val in enumerate(params):
            self._param_table.setItem(row, 2 + ci, QTableWidgetItem(str(val)))

        # Connect combo
        dist_combo.currentTextChanged.connect(
            lambda text, r=row: self._on_dist_type_changed(r, text)
        )

    def _add_param_row(self):
        """Add a new empty parameter row."""
        row = self._param_table.rowCount()
        self._param_table.insertRow(row)

        # Default name
        self._param_table.setItem(row, 0, QTableWidgetItem(f"param_{row + 1}"))

        # Distribution combo
        dist_combo = QComboBox()
        dist_combo.addItems(DIST_TYPES)
        dist_combo.setCurrentText("Normal")
        self._param_table.setCellWidget(row, 1, dist_combo)

        # Default params for Normal
        self._param_table.setItem(row, 2, QTableWidgetItem("0.0"))
        self._param_table.setItem(row, 3, QTableWidgetItem("1.0"))
        self._param_table.setItem(row, 4, QTableWidgetItem(""))

        dist_combo.currentTextChanged.connect(
            lambda text, r=row: self._on_dist_type_changed(r, text)
        )

        # Set column 3 header hint
        self._update_param_headers(row, "Normal")

    def _remove_param_row(self):
        row = self._param_table.currentRow()
        if row >= 0:
            self._param_table.removeRow(row)

    def _on_dist_type_changed(self, row: int, dist_type: str):
        """Update param column headers and defaults when dist type changes."""
        self._update_param_headers(row, dist_type)

    def _update_param_headers(self, row: int, dist_type: str):
        """Update the column headers for the parameter row."""
        meta = DIST_PARAMS_META.get(dist_type, [("Param 1", 0.0), ("Param 2", 1.0), ("", 0.0)])
        headers = ["Param 1", "Param 2", "Param 3"]
        for i, (label, default) in enumerate(meta):
            headers[i] = label

        self._param_table.setHorizontalHeaderLabels([
            "Parameter", "Distribution", headers[0], headers[1], headers[2]
        ])

    def _get_distributions(self) -> Tuple[Dict[str, Distribution], Dict[str, Any]]:
        """Parse distributions from the table. Returns (dist_dict, base_values)."""
        dist_dict: Dict[str, Distribution] = {}
        base_values: Dict[str, float] = {}

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            dist_widget = self._param_table.cellWidget(row, 1)

            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue

            dist_type = (
                dist_widget.currentText() if isinstance(dist_widget, QComboBox)
                else "Normal"
            )

            # Read param values
            param_values = []
            for ci in range(3):
                item = self._param_table.item(row, 2 + ci)
                if item and item.text().strip():
                    try:
                        param_values.append(float(item.text().strip()))
                    except ValueError:
                        param_values.append(0.0)

            try:
                if dist_type == "Normal" and len(param_values) >= 2:
                    dist = Normal(param_values[0], max(param_values[1], 1e-6))
                    base_values[name] = param_values[0]
                elif dist_type == "LogNormal" and len(param_values) >= 2:
                    dist = LogNormal(param_values[0], max(param_values[1], 1e-6))
                    base_values[name] = param_values[0]
                elif dist_type == "Uniform" and len(param_values) >= 2:
                    a, b = param_values[0], param_values[1]
                    if b <= a:
                        b = a + 1e-6
                    dist = Uniform(a, b)
                    base_values[name] = (a + b) / 2.0
                elif dist_type == "Triangular" and len(param_values) >= 3:
                    a, mode, b = param_values[0], param_values[1], param_values[2]
                    if b <= a:
                        b = a + 1e-6
                    mode = max(a, min(b, mode))
                    dist = Triangular(a, mode, b)
                    base_values[name] = mode
                elif dist_type == "Beta" and len(param_values) >= 2:
                    dist = Beta(max(param_values[0], 1e-6),
                                max(param_values[1], 1e-6))
                    base_values[name] = param_values[0] / (param_values[0] + param_values[1]) if (param_values[0] + param_values[1]) > 0 else 0.5
                else:
                    continue

                dist_dict[name] = dist
            except Exception:
                continue

        return dist_dict, base_values

    def _run_simulation(self):
        dist_dict, base_values = self._get_distributions()
        if not dist_dict:
            QMessageBox.warning(self, "No Parameters",
                                "At least one valid parameter distribution is required.")
            return

        n_samples = self._n_samples_spin.value()
        conf_level = self._conf_spin.value()
        seed = self._seed_spin.value()
        if seed < 0:
            seed = None

        mc_input = MCInput(
            parameters=dist_dict,
            model_function=self.model_function,
            output_keys=self._output_keys,
            n_samples=n_samples,
            seed=seed,
            confidence_level=conf_level,
        )

        # Disable UI
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._status_label.setText(f"Running {n_samples} Monte Carlo samples...")
        self._stats_table.setRowCount(0)
        self._conv_text.clear()

        self._worker = _MCWorker(mc_input)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel_simulation(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.terminate()
            self._worker.wait(2000)
            self._status_label.setText("Simulation cancelled.")
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, n_done: int, n_total: int, status: str):
        pct = int(n_done / max(1, n_total) * 100)
        self._progress_bar.setValue(pct)

    def _on_finished(self, result: MCResult):
        self.mc_result = result
        self._progress_bar.setValue(100)

        n_out = result.n_samples
        elapsed = result.elapsed_time
        self._status_label.setText(
            f"Done. {n_out} valid outputs in {elapsed:.2f}s."
        )

        # Populate statistics table
        self._stats_table.setRowCount(0)
        for key, stats_dict in result.statistics.items():
            key_display = str(key) if key is not None else "output"
            row = self._stats_table.rowCount()
            self._stats_table.insertRow(row)
            self._stats_table.setItem(row, 0, QTableWidgetItem(key_display))
            self._stats_table.setItem(row, 1, QTableWidgetItem(f"{stats_dict['mean']:.4g}"))
            self._stats_table.setItem(row, 2, QTableWidgetItem(f"{stats_dict['std']:.4g}"))
            self._stats_table.setItem(row, 3, QTableWidgetItem(f"{stats_dict['p50']:.4g}"))
            self._stats_table.setItem(row, 4, QTableWidgetItem(f"{stats_dict['p5']:.4g}"))
            self._stats_table.setItem(row, 5, QTableWidgetItem(f"{stats_dict['p95']:.4g}"))
            self._stats_table.setItem(row, 6, QTableWidgetItem(
                f"[{stats_dict['ci_low']:.4g}, {stats_dict['ci_high']:.4g}]"
            ))

        # Draw histogram
        self._draw_histogram(result)

        # Convergence check
        self._update_convergence(result)

        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _draw_histogram(self, result: MCResult):
        """Draw histogram of output distributions."""
        self._hist_canvas.figure.clear()

        outputs = result.outputs
        if not outputs:
            ax = self._hist_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No output data",
                     ha="center", va="center",
                     transform=ax.transAxes)
            self._hist_canvas.draw()
            return

        n_plots = min(len(outputs), 4)
        n_cols = min(n_plots, 2)
        n_rows = (n_plots + 1) // 2

        for idx, (key, arr) in enumerate(outputs.items()):
            if idx >= 4:
                break
            ax = self._hist_canvas.figure.add_subplot(n_rows, n_cols, idx + 1)

            ax.hist(arr, bins=min(50, max(10, len(arr) // 10)),
                     density=True, alpha=0.7, color="#1976D2",
                     edgecolor="white", linewidth=0.5)

            # Add KDE if SciPy available
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                x_range = np.linspace(float(np.min(arr)), float(np.max(arr)), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.8)
            except Exception:
                pass

            # Add mean and CI lines
            if key in result.statistics:
                s = result.statistics[key]
                ax.axvline(s["mean"], color="red", linestyle="--", linewidth=1.5,
                            label=f"μ={s['mean']:.4g}")
                ax.axvline(s["p5"], color="green", linestyle=":", linewidth=1, alpha=0.7)
                ax.axvline(s["p95"], color="green", linestyle=":", linewidth=1, alpha=0.7)
                ax.axvline(s["ci_low"], color="orange", linestyle="-.", linewidth=1,
                            alpha=0.7)
                ax.axvline(s["ci_high"], color="orange", linestyle="-.", linewidth=1,
                            alpha=0.7)

            ax.set_title(str(key) if key is not None else "Output", fontsize=10)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)

        self._hist_canvas.figure.tight_layout()
        self._hist_canvas.draw()

    def _update_convergence(self, result: MCResult):
        """Display convergence information."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append("Convergence Assessment")
        lines.append(f"{'='*60}")

        converged = convergence_check(result)
        lines.append(f"Converged (CI width < 10% of mean): {'YES' if converged else 'NO'}")

        for key, stats_dict in result.statistics.items():
            key_display = str(key) if key is not None else "output"
            ci_width = stats_dict["ci_high"] - stats_dict["ci_low"]
            mean_val = stats_dict["mean"]
            rel_ci = ci_width / abs(mean_val) if abs(mean_val) > 1e-12 else float("inf")
            lines.append(
                f"  {key_display}: CI width = {ci_width:.4g}, "
                f"mean = {mean_val:.4g}, relative = {rel_ci:.2%}"
            )

            # Estimate required samples
            n_est = estimate_required_samples(result, key, target_ci_width=0.05)
            if n_est > result.n_samples:
                lines.append(
                    f"    → Estimated {n_est} samples needed for 5% CI width."
                )
            else:
                lines.append(
                    f"    → Current sample size ({result.n_samples}) "
                    f"sufficient for 5% CI width."
                )

        # Correlations
        if result.correlations:
            lines.append("")
            lines.append("Input-Output Correlations:")
            for out_key, corr_dict in result.correlations.items():
                out_display = str(out_key) if out_key is not None else "output"
                lines.append(f"  {out_display}:")
                for param, corr in sorted(
                    corr_dict.items(), key=lambda x: abs(x[1]), reverse=True
                ):
                    lines.append(f"    {param}: {corr:+.4f}")

        self._conv_text.setText("\n".join(lines))

    def set_parameters(self, parameters: Dict[str, Distribution]):
        """Set or update the parameter distributions."""
        self._input_params = parameters
        self._param_table.setRowCount(0)
        self._populate_params()

    def set_output_keys(self, keys: List[str]):
        """Set the output metric keys to track."""
        self._output_keys = keys
