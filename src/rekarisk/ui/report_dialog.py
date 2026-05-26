"""
Rekarisk UI — Report Dialog.

PyQt6 dialog for configuring and generating consequence analysis reports.
Supports:
    - Report section selection (cover, TOC, summary, input, results, QRA, conclusion, appendix)
    - Cover page customization (project name, author, date, version, organization)
    - Output format selection (PDF, Excel, CSV, JSON, GeoJSON, KML)
    - Output directory selector
    - Progress indicator during generation
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QTextEdit, QCheckBox, QPushButton,
    QFileDialog, QProgressBar, QFormLayout, QTabWidget,
    QWidget, QScrollArea, QMessageBox, QDialogButtonBox,
    QSplitter, QSpinBox, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont


class ReportDialog(QDialog):
    """Dialog for configuring and generating a consequence analysis report.

    Signal:
        report_generated: Emitted with the output directory path on success.
    """

    report_generated = pyqtSignal(str)

    def __init__(
        self,
        project_data: Dict[str, Any],
        results: List[Dict[str, Any]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._project_data = project_data
        self._results = results

        self.setWindowTitle("Generate Report — Rekarisk")
        self.setMinimumSize(700, 600)
        self.resize(750, 650)

        self._setup_ui()
        self._load_defaults()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Tab widget ──
        self.tabs = QTabWidget()

        self._cover_tab = _CoverTab(self._project_data)
        self._sections_tab = _SectionsTab()
        self._formats_tab = _FormatsTab()
        self._summary_tab = _SummaryTextTab()

        self.tabs.addTab(self._cover_tab, "📄 Cover Page")
        self.tabs.addTab(self._sections_tab, "📑 Sections")
        self.tabs.addTab(self._formats_tab, "💾 Output")
        self.tabs.addTab(self._summary_tab, "📝 Text")

        layout.addWidget(self.tabs)

        # ── Output dir ──
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output Directory:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self.dir_edit, 1)

        self.dir_button = QPushButton("Browse...")
        self.dir_button.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(self.dir_button)
        layout.addLayout(dir_layout)

        # ── Progress bar ──
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.preview_btn = QPushButton("Preview Settings")
        self.preview_btn.clicked.connect(self._preview_settings)
        btn_layout.addWidget(self.preview_btn)

        self.generate_btn = QPushButton("🚀 Generate")
        self.generate_btn.setMinimumWidth(120)
        self.generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self.generate_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _load_defaults(self):
        """Load default values from project data."""
        proj_name = self._project_data.get("name", "Untitled")
        self._cover_tab.project_name_edit.setText(proj_name)
        self._cover_tab.date_edit.setText(datetime.now().strftime("%d %B %Y"))
        self._cover_tab.version_edit.setText(
            self._project_data.get("format_version", "1.0")
        )

        # Default output dir
        default_dir = str(Path.home() / "Documents" / "Rekarisk_Reports")
        self.dir_edit.setText(default_dir)

    def _browse_output_dir(self):
        """Open directory chooser."""
        current = self.dir_edit.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current,
        )
        if directory:
            self.dir_edit.setText(directory)

    def _get_selected_formats(self) -> List[str]:
        """Return list of selected output formats."""
        formats = []
        fmt_checks = [
            ("pdf", self._formats_tab.pdf_check),
            ("excel", self._formats_tab.excel_check),
            ("csv", self._formats_tab.csv_check),
            ("json", self._formats_tab.json_check),
            ("geojson", self._formats_tab.geojson_check),
            ("kml", self._formats_tab.kml_check),
            ("png", self._formats_tab.png_check),
            ("txt", self._formats_tab.txt_check),
        ]
        for name, check in fmt_checks:
            if check.isChecked():
                formats.append(name)
        return formats

    def _preview_settings(self):
        """Show a summary of selected export settings."""
        sections = self._sections_tab.get_selected()
        formats = self._get_selected_formats()
        output_dir = self.dir_edit.text()

        msg = (
            f"<b>Output Directory:</b> {output_dir}<br><br>"
            f"<b>Formats:</b> {', '.join(f.upper() for f in formats) or 'None'}<br><br>"
            f"<b>Sections:</b><br>"
            + "<br>".join(f"&nbsp;&nbsp;✓ {s}" for s in sections)
        )

        QMessageBox.information(self, "Report Settings Preview", msg)

    def _on_generate(self):
        """Start report generation."""
        output_dir = self.dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Missing Directory",
                                "Please select an output directory.")
            return

        formats = self._get_selected_formats()
        if not formats:
            QMessageBox.warning(self, "No Formats",
                                "Please select at least one output format.")
            return

        # Create output dir
        os.makedirs(output_dir, exist_ok=True)

        self.progress.setVisible(True)
        self.progress.setMaximum(len(formats))
        self.progress.setValue(0)

        try:
            generated_files = self._run_export(output_dir, formats)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self.progress.setVisible(False)
            return

        self.progress.setVisible(False)

        # Summary
        msg = "<b>Report generation complete!</b><br><br>Generated files:<br>"
        msg += "<br>".join(f"&nbsp;&nbsp;📄 {f}" for f in generated_files)

        QMessageBox.information(self, "Report Generated", msg)
        self.report_generated.emit(output_dir)
        self.accept()

    def _run_export(self, output_dir: str, formats: List[str]) -> List[str]:
        """Execute the selected export operations.

        Returns
        -------
        list[str]
            Paths of generated files.
        """
        from rekarisk.report.pdf_generator import generate_report
        from rekarisk.report.excel_export import export_to_excel
        from rekarisk.report.text_export import export_csv, export_json, export_summary_text
        from rekarisk.report.gis_export import contours_to_geojson, contours_to_kml
        from rekarisk.report.image_export import export_all_plots

        project_name = self._project_data.get("name", "report")
        safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in project_name).strip()[:50]
        if not safe_name:
            safe_name = "rekarisk_report"

        cover_info = self._cover_tab.get_info()
        sections = {
            "cover": self._sections_tab.cover_check.isChecked(),
            "toc": self._sections_tab.toc_check.isChecked(),
            "summary": self._sections_tab.summary_check.isChecked(),
            "input": self._sections_tab.input_check.isChecked(),
            "results": self._sections_tab.results_check.isChecked(),
            "qra": self._sections_tab.qra_check.isChecked(),
            "conclusion": self._sections_tab.conclusion_check.isChecked(),
            "appendix": self._sections_tab.appendix_check.isChecked(),
        }

        executive_summary = self._summary_tab.exec_edit.toPlainText()
        conclusion = self._summary_tab.conclusion_edit.toPlainText()

        generated: List[str] = []
        progress_val = 0

        for fmt in formats:
            try:
                if fmt == "pdf":
                    path = os.path.join(output_dir, f"{safe_name}.pdf")
                    generate_report(
                        self._project_data, self._results, path,
                        sections=sections, cover_info=cover_info,
                        executive_summary=executive_summary,
                        conclusion=conclusion,
                    )
                    generated.append(path)

                elif fmt == "excel":
                    path = os.path.join(output_dir, f"{safe_name}.xlsx")
                    export_to_excel(self._project_data, self._results, path)
                    generated.append(path)

                elif fmt == "csv":
                    path = os.path.join(output_dir, f"{safe_name}.csv")
                    export_csv(self._results, path)
                    generated.append(path)

                elif fmt == "json":
                    path = os.path.join(output_dir, f"{safe_name}.json")
                    export_json(self._project_data, self._results, path)
                    generated.append(path)

                elif fmt == "geojson":
                    path = os.path.join(output_dir, f"{safe_name}.geojson")
                    # Aggregate contour data from all results
                    contour_list = []
                    for res in self._results:
                        grid = res.get("grid_data", {})
                        if "x" in grid and "y" in grid and "Z" in grid:
                            contour_list.append({
                                "x": grid["x"],
                                "y": grid["y"],
                                "Z": grid["Z"],
                                "levels": grid.get("levels", []),
                                "scenario": res.get("name", ""),
                                "substance": res.get("inputs", {}).get("substance", ""),
                            })
                    if contour_list:
                        contours_to_geojson(contour_list, crs="local", output_path=path)
                    else:
                        # Write empty FeatureCollection
                        import json
                        with open(path, "w") as f:
                            json.dump({"type": "FeatureCollection", "features": []}, f)
                    generated.append(path)

                elif fmt == "kml":
                    path = os.path.join(output_dir, f"{safe_name}.kml")
                    contour_list = []
                    for res in self._results:
                        grid = res.get("grid_data", {})
                        if "x" in grid and "y" in grid and "Z" in grid:
                            contour_list.append({
                                "x": grid["x"],
                                "y": grid["y"],
                                "Z": grid["Z"],
                                "levels": grid.get("levels", []),
                                "scenario": res.get("name", ""),
                            })
                    contours_to_kml(contour_list, output_path=path, name=project_name)
                    generated.append(path)

                elif fmt == "png":
                    plots_dir = os.path.join(output_dir, "plots")
                    plot_paths = export_all_plots(self._results, plots_dir, format="png")
                    generated.extend(plot_paths)

                elif fmt == "txt":
                    path = os.path.join(output_dir, f"{safe_name}.txt")
                    export_summary_text(self._project_data, self._results, path)
                    generated.append(path)

            except Exception as e:
                generated.append(f"[FAILED:{fmt}] {str(e)[:60]}")

            progress_val += 1
            self.progress.setValue(progress_val)

        return generated


# ══════════════════════════════════════════════════════════════════════════════
# Tab Widgets
# ══════════════════════════════════════════════════════════════════════════════

class _CoverTab(QWidget):
    """Cover page configuration tab."""

    def __init__(self, project_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setSpacing(10)

        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("e.g., LPG Storage Tank QRA")
        layout.addRow("Project Name:", self.project_name_edit)

        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("e.g., Arie Nugraha")
        layout.addRow("Author:", self.author_edit)

        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("e.g., 26 May 2026")
        layout.addRow("Date:", self.date_edit)

        self.version_edit = QLineEdit()
        self.version_edit.setPlaceholderText("e.g., 1.0")
        layout.addRow("Version:", self.version_edit)

        self.org_edit = QLineEdit()
        self.org_edit.setPlaceholderText("e.g., PT Reka Engineering")
        layout.addRow("Organization:", self.org_edit)

        # Spacer
        layout.addRow("", QLabel(""))

        info_label = QLabel(
            "These fields appear on the report cover page. "
            "Leave blank to omit."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addRow(info_label)

    def get_info(self) -> Dict[str, str]:
        """Return cover info dict."""
        return {
            "project_name": self.project_name_edit.text(),
            "author": self.author_edit.text(),
            "date": self.date_edit.text(),
            "version": self.version_edit.text(),
            "organization": self.org_edit.text(),
        }


class _SectionsTab(QWidget):
    """Report section selection tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Select sections to include in the report:")
        title.setFont(QFont("", -1, QFont.Weight.Bold))
        layout.addWidget(title)

        self.cover_check = QCheckBox("📄 Cover Page")
        self.cover_check.setChecked(True)
        layout.addWidget(self.cover_check)

        self.toc_check = QCheckBox("📑 Table of Contents")
        self.toc_check.setChecked(True)
        layout.addWidget(self.toc_check)

        self.summary_check = QCheckBox("📋 Executive Summary")
        self.summary_check.setChecked(True)
        layout.addWidget(self.summary_check)

        self.input_check = QCheckBox("📊 Input Data (weather, substances, parameters)")
        self.input_check.setChecked(True)
        layout.addWidget(self.input_check)

        self.results_check = QCheckBox("📈 Results (per scenario: tables, plots, thresholds)")
        self.results_check.setChecked(True)
        layout.addWidget(self.results_check)

        self.qra_check = QCheckBox("⚠️ QRA Section (IR contours, FN curve, risk matrix)")
        self.qra_check.setChecked(True)
        layout.addWidget(self.qra_check)

        self.conclusion_check = QCheckBox("✅ Conclusion & Recommendations")
        self.conclusion_check.setChecked(True)
        layout.addWidget(self.conclusion_check)

        self.appendix_check = QCheckBox("📚 Appendix (methodology, references, assumptions)")
        self.appendix_check.setChecked(True)
        layout.addWidget(self.appendix_check)

        layout.addStretch()

    def get_selected(self) -> List[str]:
        """Return list of selected section names."""
        sections = []
        mapping = [
            ("Cover Page", self.cover_check),
            ("Table of Contents", self.toc_check),
            ("Executive Summary", self.summary_check),
            ("Input Data", self.input_check),
            ("Results", self.results_check),
            ("QRA Section", self.qra_check),
            ("Conclusion", self.conclusion_check),
            ("Appendix", self.appendix_check),
        ]
        for name, check in mapping:
            if check.isChecked():
                sections.append(name)
        return sections


class _FormatsTab(QWidget):
    """Output format selection tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Select output formats:")
        title.setFont(QFont("", -1, QFont.Weight.Bold))
        layout.addWidget(title)

        self.pdf_check = QCheckBox("📕 PDF Report (full formatted document)")
        self.pdf_check.setChecked(True)
        layout.addWidget(self.pdf_check)

        self.excel_check = QCheckBox("📗 Excel Workbook (.xlsx, multi-sheet)")
        self.excel_check.setChecked(True)
        layout.addWidget(self.excel_check)

        self.csv_check = QCheckBox("📄 CSV Tables (results tables)")
        self.csv_check.setChecked(True)
        layout.addWidget(self.csv_check)

        self.json_check = QCheckBox("📦 JSON Export (complete project + results)")
        self.json_check.setChecked(True)
        layout.addWidget(self.json_check)

        self.geojson_check = QCheckBox("🗺️ GeoJSON Overlay (GIS contour data)")
        self.geojson_check.setChecked(False)
        layout.addWidget(self.geojson_check)

        self.kml_check = QCheckBox("🌍 KML Overlay (Google Earth)")
        self.kml_check.setChecked(False)
        layout.addWidget(self.kml_check)

        self.png_check = QCheckBox("🖼️ PNG Images (contour plots, FN curve, risk matrix)")
        self.png_check.setChecked(False)
        layout.addWidget(self.png_check)

        self.txt_check = QCheckBox("📝 Text Summary (human-readable)")
        self.txt_check.setChecked(False)
        layout.addWidget(self.txt_check)

        layout.addStretch()


class _SummaryTextTab(QWidget):
    """Executive summary and conclusion text tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Executive Summary:"))
        self.exec_edit = QTextEdit()
        self.exec_edit.setPlaceholderText(
            "Enter a brief executive summary of the analysis...\n\n"
            "This appears in the PDF report before detailed results."
        )
        self.exec_edit.setMaximumHeight(120)
        layout.addWidget(self.exec_edit)

        layout.addWidget(QLabel("Conclusion & Recommendations:"))
        self.conclusion_edit = QTextEdit()
        self.conclusion_edit.setPlaceholderText(
            "Enter conclusions and recommendations...\n\n"
            "This appears at the end of the PDF report."
        )
        self.conclusion_edit.setMaximumHeight(120)
        layout.addWidget(self.conclusion_edit)

        layout.addStretch()
