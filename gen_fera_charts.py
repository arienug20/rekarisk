#!/usr/bin/env python3
"""Generate comparison charts for FERA NKT study: PHAST vs Rekarisk."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 200,
})

OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario'

# ── Data from comparison results ──

# Release rates (Fullbore only for gas phase)
iso_labels = ['ISO 1\nFullbore', 'ISO 2G\nFullbore', 'ISO 3G\nFullbore', 'ISO 5\nFullbore']
phast_rates = [108.671, 107.610, 105.270, 246.480]
rr_rates = [118.329, 118.329, 115.857, 113.381]

# Jet Fire flame lengths
jf_labels = ['ISO 1', 'ISO 2G', 'ISO 3G', 'ISO 5']
phast_flame = [84.8, 84.6, 84.1, 84.6]
rr_flame = [93.3, 93.3, 92.5, 91.7]

# Jet fire distances (4.73 kW/m² threshold, wind 1.35C)
phast_jf_dist = [135.7, 135.4, 134.7, 135.4]
rr_jf_dist = [122.7, 122.7, 121.3, 119.9]

# Flash fire (50% LFL)
ff_labels = ['ISO 1\n1.35C', 'ISO 1\n1.81C', 'ISO 1\n5.5D',
             'ISO 2G\n1.35C', 'ISO 2G\n1.81C', 'ISO 2G\n5.5D']
phast_ff = [45.3, 45.7, 44.9, 47.1, 47.5, 46.7]
rr_ff = [76.1, 71.8, 57.5, 76.1, 71.8, 57.5]

# VCE (0.35 bar)
vce_labels = ['ISO 1\n1.35C', 'ISO 1\n1.81C', 'ISO 1\n5.5D',
              'ISO 2G\n1.35C', 'ISO 2G\n1.81C', 'ISO 2G\n5.5D']
phast_vce = [34.8, 34.7, 34.6, 65.3, 65.4, 65.1]
rr_vce = [29.4, 29.4, 29.4, 48.9, 48.9, 48.9]


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1: Release Rate Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(iso_labels))
w = 0.35
bars1 = ax.bar(x - w/2, phast_rates, w, label='PHAST v9.0', color='#2196F3', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + w/2, rr_rates, w, label='Rekarisk', color='#FF9800', edgecolor='white', linewidth=1.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2196F3')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#FF9800')

ax.set_ylabel('Initial Release Rate (kg/s)', fontweight='bold')
ax.set_title('FERA NKT — Release Rate Comparison\n(Fullbore, Gas Phase)', fontweight='bold', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(iso_labels, fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(phast_rates + rr_rates) * 1.2)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_01_release_rate.png', dpi=200)
plt.close()
print('Chart 1 saved')

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2: Jet Fire Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Flame length
x = np.arange(len(jf_labels))
bars1 = ax1.bar(x - w/2, phast_flame, w, label='PHAST', color='#2196F3', edgecolor='white')
bars2 = ax1.bar(x + w/2, rr_flame, w, label='Rekarisk', color='#FF9800', edgecolor='white')
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#2196F3')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#FF9800')
ax1.set_ylabel('Flame Length (m)', fontweight='bold')
ax1.set_title('Jet Fire Flame Length', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(jf_labels, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Distance to 4.73 kW/m²
bars1 = ax2.bar(x - w/2, phast_jf_dist, w, label='PHAST', color='#2196F3', edgecolor='white')
bars2 = ax2.bar(x + w/2, rr_jf_dist, w, label='Rekarisk', color='#FF9800', edgecolor='white')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#2196F3')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold', color='#FF9800')
ax2.set_ylabel('Distance (m)', fontweight='bold')
ax2.set_title('Jet Fire Distance to 4.73 kW/m²', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(jf_labels, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('FERA NKT — Jet Fire Comparison (Fullbore, Wind 1.35C)', fontweight='bold', fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_02_jetfire.png', dpi=200, bbox_inches='tight')
plt.close()
print('Chart 2 saved')

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3: Flash Fire & VCE
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(ff_labels))
bars1 = ax1.bar(x - w/2, phast_ff, w, label='PHAST', color='#2196F3', edgecolor='white')
bars2 = ax1.bar(x + w/2, rr_ff, w, label='Rekarisk', color='#FF9800', edgecolor='white')
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold', color='#2196F3')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold', color='#FF9800')
ax1.set_ylabel('Distance to 50% LFL (m)', fontweight='bold')
ax1.set_title('Flash Fire Distance', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(ff_labels, fontsize=10, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

x = np.arange(len(vce_labels))
bars1 = ax2.bar(x - w/2, phast_vce, w, label='PHAST', color='#2196F3', edgecolor='white')
bars2 = ax2.bar(x + w/2, rr_vce, w, label='Rekarisk', color='#FF9800', edgecolor='white')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold', color='#2196F3')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold', color='#FF9800')
ax2.set_ylabel('Distance to 0.35 bar (m)', fontweight='bold')
ax2.set_title('VCE Overpressure Distance', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(vce_labels, fontsize=10, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('FERA NKT — Flash Fire & VCE Comparison (Fullbore)', fontweight='bold', fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_03_flashfire_vce.png', dpi=200, bbox_inches='tight')
plt.close()
print('Chart 3 saved')

# ══════════════════════════════════════════════════════════════════════════════
# Chart 4: Summary Radar / Deviation
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

categories = [
    'Release Rate\n(Fullbore Gas)',
    'Jet Fire\nFlame Length',
    'Jet Fire\n4.73 kW/m²',
    'Jet Fire\n6.3 kW/m²',
    'Jet Fire\n12.5 kW/m²',
    'Flash Fire\n50% LFL',
    'VCE\n0.35 bar',
    'VCE\n0.5 bar',
]
# Average deviation percentages (Rekarisk - PHAST) / PHAST * 100
deviations = [
    np.mean([8.9, 10.0, 10.1, -54.0]),  # release rate fullbore gas (exclude ISO 5 anomaly)
    np.mean([9.9, 10.2, 9.9, 8.4]),     # flame length
    np.mean([-9.6, -9.4, -9.9, -11.4]), # 4.73
    np.mean([-19.0, -18.9, -19.3, -20.7]), # 6.3
    np.mean([-40.3, -40.2, -40.6, -41.6]), # 12.5
    np.mean([67.9, 57.2, 27.8, 61.6, 51.2, 23.0]), # flash fire
    np.mean([-15.3, -15.2, -14.9, -25.1, -25.1, -24.9]), # VCE 0.35
    np.mean([-30.5, -30.5, -30.2, -39.2, -39.2, -39.0]), # VCE 0.5
]

colors = ['#4CAF50' if abs(d) <= 20 else '#FFC107' if abs(d) <= 40 else '#F44336' for d in deviations]
bars = ax.barh(range(len(categories)), deviations, color=colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1.5)
ax.axvline(x=-20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(x=20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(x=-40, color='#FFC107', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(x=40, color='#FFC107', linewidth=1, linestyle='--', alpha=0.5)

for bar, dev in zip(bars, deviations):
    x_pos = bar.get_width() + (2 if dev >= 0 else -2)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{dev:+.0f}%', va='center', ha='left' if dev >= 0 else 'right',
            fontsize=12, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Good (within ±20%)'),
    Patch(facecolor='#FFC107', label='Fair (±20-40%)'),
    Patch(facecolor='#F44336', label='Needs calibration (>±40%)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
ax.set_xlabel('Deviation from PHAST (%)', fontweight='bold')
ax.set_title('FERA NKT — Rekarisk vs PHAST Deviation Summary\n(Positive = Rekarisk higher, Negative = Rekarisk lower)',
             fontweight='bold', fontsize=14)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_04_deviation_summary.png', dpi=200, bbox_inches='tight')
plt.close()
print('Chart 4 saved')

print('\nAll charts generated successfully!')
