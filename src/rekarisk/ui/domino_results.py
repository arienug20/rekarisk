"""
Rekarisk UI — Domino Analysis Results Panel.

Displays domino / escalation analysis results:
  - Escalation map (facility layout with arrows)
  - Summary table (target, distance, vector, intensity, damage, probability, TTF)
  - Domino chain diagram
  - Escalation probability bar chart
  - CSV export
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QSplitter,
    QTextBrowser, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    import matplotlib
    matplotlib.use("QtAgg")


# ══════════════════════════════════════════════════════════════════════════════
# Results Panel
# ══════════════════════════════════════════════════════════════════════════════

class DominoResultsPanel(QWidget):
    """Displays domino analysis results with interactive plots and tables."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._result = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Summary label
        self._summary_label = QLabel("Run domino analysis to see results.")
        self._summary_label.setFont(QFont("Sans", 11))
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("padding: 8px; background: #ebf5fb; border-radius: 4px;")
        layout.addWidget(self._summary_label)

        # Tabs for different views
        self._tabs = QTabWidget()

        # Tab 1: Escalation Map
        self._map_tab = QWidget()
        map_layout = QVBoxLayout(self._map_tab)
        if HAS_MPL:
            self._map_figure = Figure(figsize=(10, 8), dpi=120)
            self._map_canvas = FigureCanvas(self._map_figure)
            map_layout.addWidget(self._map_canvas)
        else:
            map_layout.addWidget(QLabel("matplotlib required for plots"))
        self._tabs.addTab(self._map_tab, "🗺️ Escalation Map")

        # Tab 2: Escalation Summary Chart
        self._summary_tab = QWidget()
        summary_layout = QVBoxLayout(self._summary_tab)
        if HAS_MPL:
            self._summary_figure = Figure(figsize=(10, 6), dpi=120)
            self._summary_canvas = FigureCanvas(self._summary_figure)
            summary_layout.addWidget(self._summary_canvas)
        self._tabs.addTab(self._summary_tab, "📊 Escalation Summary")

        # Tab 3: Chain Diagram
        self._chain_tab = QWidget()
        chain_layout = QVBoxLayout(self._chain_tab)
        if HAS_MPL:
            self._chain_figure = Figure(figsize=(12, 8), dpi=120)
            self._chain_canvas = FigureCanvas(self._chain_figure)
            chain_layout.addWidget(self._chain_canvas)
        self._tabs.addTab(self._chain_tab, "⛓️ Chain Diagram")

        # Tab 4: Data Table
        self._table_tab = QWidget()
        table_layout = QVBoxLayout(self._table_tab)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Target", "Distance (m)", "Vector", "Intensity",
            "Damage Level", "P(escalation)", "TTF (min)", "Released (kg)",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        table_layout.addWidget(self._table)

        # Export button
        export_btn = QPushButton("📄 Export to CSV")
        export_btn.clicked.connect(self._export_csv)
        table_layout.addWidget(export_btn)

        self._tabs.addTab(self._table_tab, "📋 Data Table")

        layout.addWidget(self._tabs)

    def set_result(self, result: Any):
        """Display domino analysis results.

        Parameters
        ----------
        result : DominoAnalysisResult
            The result from run_domino_analysis().
        """
        self._result = result

        # Update summary label
        s = result.summary
        self._summary_label.setText(
            f"<b>Domino Analysis Complete</b><br/>"
            f"Equipment at risk: <b>{len(s.get('equipment_at_risk', []))}</b> of {s.get('total_equipment', 0)} &nbsp;|&nbsp; "
            f"Significant links: <b>{s.get('significant_links', 0)}</b> &nbsp;|&nbsp; "
            f"Domino scenarios: <b>{s.get('domino_scenarios', 0)}</b> &nbsp;|&nbsp; "
            f"Max cascade: <b>Order {s.get('max_cascade_order', 1)}</b> &nbsp;|&nbsp; "
            f"Max distance: <b>{s.get('max_escalation_distance_m', 0):.0f} m</b>"
        )

        # Draw plots
        self._draw_escalation_map(result)
        self._draw_summary_chart(result)
        self._draw_chain_diagram(result)
        self._fill_table(result)

        # Switch to map tab
        self._tabs.setCurrentIndex(0)

    # ── Drawing methods ──

    def _draw_escalation_map(self, result):
        if not HAS_MPL:
            return
        from rekarisk.models.qra.domino import (
            EquipmentType, EscalationVector, DamageLevel,
        )

        fig = self._map_figure
        fig.clear()
        ax = fig.add_subplot(111)

        eq_map = {eq.id: eq for eq in result.equipment_list}
        at_risk = set(result.summary.get("equipment_at_risk", []))

        eq_colors = {
            EquipmentType.ATMOSPHERIC_TANK: "#3498db",
            EquipmentType.PRESSURE_VESSEL: "#e74c3c",
            EquipmentType.COLUMN: "#1abc9c",
            EquipmentType.HEAT_EXCHANGER: "#2ecc71",
            EquipmentType.PUMP: "#e67e22",
            EquipmentType.COMPRESSOR: "#e67e22",
            EquipmentType.FIN_FAN_COOLER: "#16a085",
        }
        damage_colors = {
            DamageLevel.MODERATE: "#f1c40f",
            DamageLevel.MAJOR: "#e74c3c",
            DamageLevel.CATASTROPHIC: "#8e44ad",
        }
        vector_ls = {
            EscalationVector.THERMAL_RADIATION: "--",
            EscalationVector.OVERPRESSURE: ":",
            EscalationVector.FIRE_IMPINGEMENT: "-",
        }

        # Draw best link per unique target
        significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
        best = {}
        for link in significant:
            key = (link.source_id, link.target_id)
            if key not in best or link.escalation_prob > best[key].escalation_prob:
                best[key] = link

        for link in best.values():
            src = eq_map.get(link.source_id)
            tgt = eq_map.get(link.target_id)
            if not src or not tgt:
                continue
            color = damage_colors.get(link.damage_level, "#e74c3c")
            ls = vector_ls.get(link.vector, "-")
            lw = 1 + link.escalation_prob * 3

            ax.annotate("", xy=(tgt.x, tgt.y), xytext=(src.x, src.y),
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.4",
                            color=color, linewidth=lw, linestyle=ls,
                            connectionstyle="arc3,rad=0.08",
                        ), zorder=3)

            mx = (src.x + tgt.x) / 2
            my = (src.y + tgt.y) / 2
            unit = "kW/m²" if link.vector == EscalationVector.THERMAL_RADIATION else "kPa"
            ax.annotate(
                f"{link.intensity:.0f} {unit}\nP={link.escalation_prob:.0%}",
                (mx, my), fontsize=8, ha="center", color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
                zorder=15,
            )

        # Draw equipment
        for eq in result.equipment_list:
            c = eq_colors.get(eq.equipment_type, "#3498db")
            is_primary = eq.id == result.primary_event
            is_affected = eq.id in at_risk

            if is_primary:
                ms, ec, ew = 250, "red", 3
            elif is_affected:
                ms, ec, ew = 180, "orange", 2
            else:
                ms, ec, ew = 120, "gray", 1

            ax.scatter(eq.x, eq.y, s=ms, c=c, marker="s",
                       edgecolors=ec, linewidths=ew, zorder=10)
            ax.annotate(
                f"{eq.id}\n({eq.substance})",
                (eq.x, eq.y), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=8, fontweight="bold" if is_primary else "normal",
                color="red" if is_primary else "#2c3e50", zorder=20,
            )

        # Legend
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
                   markeredgecolor="red", markersize=10, label="Primary"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
                   markeredgecolor="orange", markersize=8, label="At Risk"),
            Patch(facecolor="#8e44ad", label="Catastrophic"),
            Patch(facecolor="#e74c3c", label="Major"),
            Patch(facecolor="#f1c40f", label="Moderate"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Domino Escalation Map — {result.primary_scenario}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._map_canvas.draw()

    def _draw_summary_chart(self, result):
        if not HAS_MPL:
            return
        from rekarisk.models.qra.domino import DamageLevel, EscalationVector

        fig = self._summary_figure
        fig.clear()
        ax = fig.add_subplot(111)

        significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
        # Best per target
        best = {}
        for l in significant:
            if l.target_id not in best or l.escalation_prob > best[l.target_id].escalation_prob:
                best[l.target_id] = l
        sorted_links = sorted(best.values(), key=lambda l: -l.escalation_prob)

        if not sorted_links:
            ax.text(0.5, 0.5, "No escalation risks", ha="center", va="center", transform=ax.transAxes)
            self._summary_canvas.draw()
            return

        targets = [l.target_id for l in sorted_links]
        probs = [l.escalation_prob * 100 for l in sorted_links]
        colors = []
        for l in sorted_links:
            if l.vector == EscalationVector.THERMAL_RADIATION:
                colors.append("#e74c3c")
            elif l.vector == EscalationVector.OVERPRESSURE:
                colors.append("#2980b9")
            else:
                colors.append("#e67e22")

        bars = ax.barh(range(len(targets)), probs, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets, fontweight="bold")
        ax.set_xlabel("Escalation Probability (%)")
        ax.set_title("Escalation Probability by Target", fontweight="bold", fontsize=12)
        ax.invert_yaxis()

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{prob:.1f}%", va="center", fontsize=9, fontweight="bold")

        legend_elements = [
            Patch(facecolor="#e74c3c", label="Thermal"),
            Patch(facecolor="#2980b9", label="Overpressure"),
            Patch(facecolor="#e67e22", label="Impingement"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        fig.tight_layout()
        self._summary_canvas.draw()

    def _draw_chain_diagram(self, result):
        if not HAS_MPL:
            return
        from rekarisk.models.qra.domino import DamageLevel

        fig = self._chain_figure
        fig.clear()
        ax = fig.add_subplot(111)

        scenarios = sorted(result.domino_scenarios, key=lambda s: -s.total_frequency)[:15]
        if not scenarios:
            ax.text(0.5, 0.5, "No domino scenarios", ha="center", va="center", transform=ax.transAxes)
            self._chain_canvas.draw()
            return

        max_chain = max(len(s.chain) for s in scenarios)

        for idx, scenario in enumerate(scenarios):
            y = idx
            for i, eq_id in enumerate(scenario.chain):
                x = i * 3.5
                is_primary = (i == 0)
                if is_primary:
                    box_c = "#e74c3c"
                else:
                    link = scenario.links[i-1] if i-1 < len(scenario.links) else None
                    if link and link.damage_level == DamageLevel.CATASTROPHIC:
                        box_c = "#8e44ad"
                    elif link and link.damage_level == DamageLevel.MAJOR:
                        box_c = "#e67e22"
                    else:
                        box_c = "#3498db"

                rect = FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=box_c, edgecolor="white", linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x, y, eq_id, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

                if i < len(scenario.chain) - 1:
                    ax.annotate("", xy=(x+2.4, y), xytext=(x+1.2, y),
                                arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.5))

            ax.text(max_chain*3.5+1, y,
                    f"f={scenario.total_frequency:.1e}/yr | {scenario.total_inventory_released:,.0f}kg",
                    va="center", fontsize=8, color="#2c3e50")

        ax.set_xlim(-1.5, max_chain*3.5+12)
        ax.set_ylim(-1, len(scenarios))
        ax.set_title("Domino Chain Diagram (Top 15)", fontweight="bold", fontsize=12)
        ax.set_yticks([])
        ax.set_xticks([i*3.5 for i in range(max_chain)])
        ax.set_xticklabels([f"Order {i+1}" for i in range(max_chain)], fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        self._chain_canvas.draw()

    def _fill_table(self, result):
        from rekarisk.models.qra.domino import DamageLevel

        significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
        # Best per unique target
        best = {}
        for l in significant:
            if l.target_id not in best or l.escalation_prob > best[l.target_id].escalation_prob:
                best[l.target_id] = l
        sorted_links = sorted(best.values(), key=lambda l: -l.escalation_prob)

        self._table.setRowCount(len(sorted_links))
        for row, link in enumerate(sorted_links):
            unit = "kW/m²" if "thermal" in link.vector.value else "kPa"
            values = [
                link.target_id,
                f"{link.distance_m:.1f}",
                link.vector.value.replace("_", " ").title(),
                f"{link.intensity:.1f} {unit}",
                link.damage_level.value.title(),
                f"{link.escalation_prob:.1%}",
                f"{link.ttf_minutes:.1f}",
                f"{link.inventory_released_kg:.0f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # Color by damage level
                if link.damage_level == DamageLevel.CATASTROPHIC:
                    item.setBackground(QColor("#f5e6f0"))
                elif link.damage_level == DamageLevel.MAJOR:
                    item.setBackground(QColor("#fce4e4"))
                self._table.setItem(row, col, item)

        self._table.resizeColumnsToContents()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Domino Results", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = []
                for col in range(self._table.columnCount()):
                    headers.append(self._table.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                for row in range(self._table.rowCount()):
                    writer.writerow([
                        self._table.item(row, col).text() if self._table.item(row, col) else ""
                        for col in range(self._table.columnCount())
                    ])
            QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
