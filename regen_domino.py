"""Regenerate domino plots with clearer fonts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

OUTDIR = Path("/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario")
DPI = 200

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,
    "savefig.dpi": DPI,
})

from rekarisk.models.qra.domino import (
    EquipmentType, SubstanceCategory, Equipment, PrimaryEvent,
    EscalationVector, DamageLevel,
    run_domino_analysis, plot_domino_map, plot_domino_chain,
)

equipment = [
    Equipment(id="TK-301", name="Propane Storage Tank", equipment_type=EquipmentType.ATMOSPHERIC_TANK,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=50000, x=0, y=0, diameter=10, height=8, operating_pressure=8, design_pressure=10, has_bund=True, bund_radius=12),
    Equipment(id="V-302", name="Propane Separator", equipment_type=EquipmentType.PRESSURE_VESSEL,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=5000, x=30, y=10, diameter=2, height=6, operating_pressure=12, design_pressure=15, is_insulated=True),
    Equipment(id="V-303", name="Butane Storage Vessel", equipment_type=EquipmentType.PRESSURE_VESSEL,
              substance="Butane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=20000, x=40, y=-15, diameter=4, height=7, operating_pressure=5, design_pressure=8),
    Equipment(id="C-304", name="De-propanizer Column", equipment_type=EquipmentType.COLUMN,
              substance="LPG Mix", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=8000, x=-20, y=25, diameter=2, height=20, operating_pressure=15, design_pressure=20, has_deluge=True),
    Equipment(id="TK-305", name="Condensate Storage", equipment_type=EquipmentType.ATMOSPHERIC_TANK,
              substance="Condensate", substance_category=SubstanceCategory.FLAMMABLE_LIQUID,
              inventory_kg=100000, x=-30, y=-20, diameter=12, height=10, operating_pressure=0.1, design_pressure=0.5, has_bund=True, bund_radius=15),
    Equipment(id="HX-306", name="Propane Chiller", equipment_type=EquipmentType.HEAT_EXCHANGER,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=2000, x=15, y=25, diameter=1, height=3, operating_pressure=10, design_pressure=14),
    Equipment(id="P-307A", name="Propane Pump A", equipment_type=EquipmentType.PUMP,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=500, x=15, y=-5, diameter=0.5, height=1.5, operating_pressure=10, design_pressure=15),
    Equipment(id="P-307B", name="Propane Pump B", equipment_type=EquipmentType.PUMP,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=500, x=20, y=5, diameter=0.5, height=1.5, operating_pressure=10, design_pressure=15),
    Equipment(id="K-308", name="Propane Compressor", equipment_type=EquipmentType.COMPRESSOR,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_GAS,
              inventory_kg=1000, x=-10, y=15, diameter=2, height=3, operating_pressure=15, design_pressure=20),
    Equipment(id="FIN-309", name="Air Cooler", equipment_type=EquipmentType.FIN_FAN_COOLER,
              substance="Propane", substance_category=SubstanceCategory.FLAMMABLE_LPG,
              inventory_kg=1500, x=5, y=-15, diameter=3, height=2, operating_pressure=12, design_pressure=16),
]

primary = PrimaryEvent(
    equipment_id="TK-301", event_type="pool_fire", frequency=5e-6,
    thermal_power_kw=150000, tnt_mass_kg=250, fireball_radius_m=50,
    source_height_m=4.0, pool_radius_m=12,
)

result = run_domino_analysis(primary_event=primary, equipment_list=equipment, max_escalation_order=3)

eq_map = {eq.id: eq for eq in result.equipment_list}

# ══════════════════════════════════════════════════════════════════════════════
# DOMINO MAP
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [1/3] Domino escalation map...")

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
    DamageLevel.NONE: "#bdc3c7", DamageLevel.MINOR: "#f1c40f",
    DamageLevel.MODERATE: "#e67e22", DamageLevel.MAJOR: "#e74c3c",
    DamageLevel.CATASTROPHIC: "#8e44ad",
}
vector_styles = {
    EscalationVector.THERMAL_RADIATION: "--",
    EscalationVector.OVERPRESSURE: ":",
    EscalationVector.FIRE_IMPINGEMENT: "-",
}

fig, ax = plt.subplots(figsize=(14, 11))

at_risk = set(result.summary.get("equipment_at_risk", []))

# Draw links first (behind equipment)
# Only draw the BEST (highest prob) link per unique (source, target) pair
best_links = {}
significant = [l for l in result.escalation_links if l.damage_level != DamageLevel.NONE]
for link in significant:
    key = (link.source_id, link.target_id)
    if key not in best_links or link.escalation_prob > best_links[key].escalation_prob:
        best_links[key] = link

for link in best_links.values():
    src = eq_map[link.source_id]
    tgt = eq_map[link.target_id]
    color = damage_colors.get(link.damage_level, "#e74c3c")
    lw = 1.5 + link.escalation_prob * 3.5
    ls = vector_styles.get(link.vector, "-")

    ax.annotate("", xy=(tgt.x, tgt.y), xytext=(src.x, src.y),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.5,head_length=0.4",
                    color=color, linewidth=lw, linestyle=ls,
                    connectionstyle="arc3,rad=0.1",
                ), zorder=3)

    # Label on midpoint
    mx = (src.x + tgt.x) / 2
    my = (src.y + tgt.y) / 2
    if link.vector == EscalationVector.THERMAL_RADIATION:
        lbl = f"{link.intensity:.0f} kW/m²"
    else:
        lbl = f"{link.intensity:.0f} kPa"
    ax.annotate(
        f"{lbl}\nP={link.escalation_prob:.0%}",
        (mx, my), fontsize=10, ha="center", color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1.5),
        zorder=15,
    )

# Draw equipment
for eq in equipment:
    color = eq_colors.get(eq.equipment_type, "#3498db")
    is_primary = eq.id == result.primary_event
    is_affected = eq.id in at_risk

    if is_primary:
        ms, ec, ew, zo = 350, "red", 4, 10
    elif is_affected:
        ms, ec, ew, zo = 250, "orange", 3, 8
    else:
        ms, ec, ew, zo = 150, "gray", 1.5, 5

    ax.scatter(eq.x, eq.y, s=ms, c=color, marker="s", edgecolors=ec, linewidths=ew, zorder=zo)
    label = f"{eq.id}\n({eq.substance})"
    ax.annotate(label, (eq.x, eq.y), textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=11, fontweight="bold" if is_primary else "normal",
                color="red" if is_primary else "#2c3e50", zorder=20)

legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markeredgecolor="red", markersize=14, label="Primary Event"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db", markeredgecolor="orange", markersize=12, label="At Risk"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db", markeredgecolor="gray", markersize=10, label="Equipment"),
    Line2D([0], [0], color="#8e44ad", linestyle="--", linewidth=2.5, label="Catastrophic"),
    Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=2.5, label="Major"),
    Line2D([0], [0], color="#e67e22", linestyle=":", linewidth=2.5, label="Moderate"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.95, edgecolor="gray")

ax.set_xlabel("X (m)", fontsize=14, fontweight="bold")
ax.set_ylabel("Y (m)", fontsize=14, fontweight="bold")
ax.set_title(f"Domino Effect Escalation Map\nPrimary: Pool Fire at TK-301 (f = 5×10⁻⁶/yr)",
             fontsize=17, fontweight="bold", pad=15)
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUTDIR / "09_domino_escalation_map.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ESCALATION SUMMARY — with clearer labels
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [2/3] Escalation summary...")

# Deduplicate: keep best link per target
best_per_target = {}
for link in significant:
    if link.target_id not in best_per_target or link.escalation_prob > best_per_target[link.target_id].escalation_prob:
        best_per_target[link.target_id] = link

sorted_links = sorted(best_per_target.values(), key=lambda l: -l.escalation_prob)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(7, len(sorted_links)*0.8)))

targets = [l.target_id for l in sorted_links]
probs = [l.escalation_prob * 100 for l in sorted_links]
intensities = [l.intensity for l in sorted_links]
vectors = [l.vector.value for l in sorted_links]

bar_colors = []
for v in vectors:
    if "thermal" in v: bar_colors.append("#e74c3c")
    elif "overpressure" in v: bar_colors.append("#2980b9")
    else: bar_colors.append("#e67e22")

y_pos = np.arange(len(targets))

# Plot 1: Escalation Probability
bars1 = ax1.barh(y_pos, probs, color=bar_colors, edgecolor="white", height=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(targets, fontsize=13, fontweight="bold")
ax1.set_xlabel("Escalation Probability (%)", fontsize=14, fontweight="bold")
ax1.set_title("Escalation Probability\nby Target Equipment", fontsize=17, fontweight="bold", pad=12)
ax1.invert_yaxis()
ax1.set_xlim(0, max(probs)*1.25 if probs else 1)

for i, (bar, prob) in enumerate(zip(bars1, probs)):
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f"{prob:.1f}%", va="center", fontsize=13, fontweight="bold", color="#2c3e50")

# Plot 2: Intensity
bars2 = ax2.barh(y_pos, intensities, color=bar_colors, edgecolor="white", height=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(targets, fontsize=13, fontweight="bold")
ax2.set_xlabel("Intensity (kW/m² or kPa)", fontsize=14, fontweight="bold")
ax2.set_title("Incident Intensity\nat Target Equipment", fontsize=17, fontweight="bold", pad=12)
ax2.invert_yaxis()
ax2.set_xlim(0, max(intensities)*1.25 if intensities else 1)

# Add unit labels
for i, (bar, link) in enumerate(zip(bars2, sorted_links)):
    unit = "kW/m²" if link.vector == EscalationVector.THERMAL_RADIATION else "kPa"
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f"{link.intensity:.1f} {unit}", va="center", fontsize=12, fontweight="bold", color="#2c3e50")

# Vector legend
legend_elements = [
    Patch(facecolor="#e74c3c", edgecolor="gray", label="Thermal Radiation"),
    Patch(facecolor="#2980b9", edgecolor="gray", label="Overpressure"),
    Patch(facecolor="#e67e22", edgecolor="gray", label="Fire Impingement"),
]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=12, framealpha=0.95)

plt.suptitle(f"Domino Escalation Summary — Pool Fire at TK-301",
             fontsize=19, fontweight="bold", y=1.02)
plt.tight_layout(w_pad=4)
fig.savefig(OUTDIR / "10_domino_summary.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# DOMINO CHAIN
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [3/3] Domino chain...")

scenarios = sorted(result.domino_scenarios, key=lambda s: -s.total_frequency)
# Only show top 20 scenarios for readability
top_scenarios = scenarios[:20]

fig, ax = plt.subplots(figsize=(16, max(8, len(top_scenarios) * 1.3)))

max_chain_len = max(len(s.chain) for s in top_scenarios)

for idx, scenario in enumerate(top_scenarios):
    y = idx
    chain = scenario.chain

    for i, eq_id in enumerate(chain):
        x = i * 4
        is_primary = (i == 0)

        if is_primary:
            box_color, text_color = "#e74c3c", "white"
        else:
            link = scenario.links[i-1] if i-1 < len(scenario.links) else None
            if link and link.damage_level == DamageLevel.CATASTROPHIC:
                box_color = "#8e44ad"
            elif link and link.damage_level == DamageLevel.MAJOR:
                box_color = "#e67e22"
            elif link and link.damage_level == DamageLevel.MODERATE:
                box_color = "#f1c40f"
            else:
                box_color = "#3498db"
            text_color = "white"

        rect = FancyBboxPatch((x-1.3, y-0.35), 2.6, 0.7,
                               boxstyle="round,pad=0.12",
                               facecolor=box_color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, eq_id, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)

        if i < len(chain) - 1:
            ax.annotate("", xy=(x+2.7, y), xytext=(x+1.3, y),
                        arrowprops=dict(arrowstyle="->", color="#7f8c8d", linewidth=2))

    # Frequency + inventory label
    ax.text(
        max_chain_len * 4 + 1.5, y,
        f"f = {scenario.total_frequency:.1e}/yr  |  {scenario.total_inventory_released:,.0f} kg released",
        va="center", fontsize=12, color="#2c3e50", fontweight="bold",
    )

ax.set_xlim(-2, max_chain_len * 4 + 14)
ax.set_ylim(-1, len(top_scenarios))
ax.set_xlabel("Escalation Order →", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("Domino Effect Chain Diagram (Top 20 Scenarios)\nPrimary: Pool Fire at TK-301",
             fontsize=17, fontweight="bold", pad=15)
ax.set_yticks([])
ax.set_xticks([i * 4 for i in range(max_chain_len)])
ax.set_xticklabels([f"Order {i+1}" for i in range(max_chain_len)], fontsize=13, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)

# Legend for colors
chain_legend = [
    Patch(facecolor="#e74c3c", edgecolor="white", label="Primary Event"),
    Patch(facecolor="#8e44ad", edgecolor="white", label="Catastrophic Damage"),
    Patch(facecolor="#e67e22", edgecolor="white", label="Major Damage"),
    Patch(facecolor="#f1c40f", edgecolor="white", label="Moderate Damage"),
]
ax.legend(handles=chain_legend, loc="lower right", fontsize=12, framealpha=0.95, edgecolor="gray")

plt.tight_layout()
fig.savefig(OUTDIR / "11_domino_chain.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("\n✅ All 3 domino plots regenerated!")
