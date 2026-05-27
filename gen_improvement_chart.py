#!/usr/bin/env python3
"""Generate improved comparison charts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'figure.dpi': 200})
OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario'

# ── Deviation summary (before vs after) ──
fig, ax = plt.subplots(figsize=(14, 7))

categories = [
    'Release Rate\n(Gas Fullbore)',
    'Jet Fire\nFlame Length',
    'Jet Fire\n4.73 kW/m²',
    'Jet Fire\n6.3 kW/m²',
    'Jet Fire\n12.5 kW/m²',
    'Jet Fire\n37.5 kW/m²',
    'Flash Fire\n50% LFL (5.5D)',
    'Liquid Release\n(Small/Medium)',
]

before = [+9.5, +9.5, -9.6, -19.0, -40.3, -72.2, +52.0, -90.0]
after  = [+9.5, +9.5, -17.8, -22.9, -36.5, -61.6, +25.0, -8.7]

x = np.arange(len(categories))
w = 0.35

bars1 = ax.bar(x - w/2, before, w, label='Before (Old Model)', color='#F44336', edgecolor='white', linewidth=1.5, alpha=0.7)
bars2 = ax.bar(x + w/2, after, w, label='After (Improved Model)', color='#4CAF50', edgecolor='white', linewidth=1.5)

for bar in bars2:
    val = bar.get_height()
    y_pos = val + (2 if val >= 0 else -4)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:+.0f}%', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=10, fontweight='bold', color='#2E7D32')

ax.axvline(x=4.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax.text(2.2, ax.get_ylim()[1]*0.9, 'JET FIRE', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

ax.axhline(y=0, color='black', linewidth=1.5)
ax.axhline(y=20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.4)
ax.axhline(y=-20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.4)
ax.axhline(y=-30, color='#FFC107', linewidth=1, linestyle='--', alpha=0.4)
ax.axhline(y=30, color='#FFC107', linewidth=1, linestyle='--', alpha=0.4)

legend_elements = [
    Patch(facecolor='#F44336', alpha=0.7, label='Before (Old Model)'),
    Patch(facecolor='#4CAF50', label='After (Improved Model)'),
    Patch(facecolor='#4CAF50', alpha=0.3, label='Good (±20%)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold', fontsize=11)
ax.set_ylabel('Deviation from PHAST (%)', fontweight='bold', fontsize=13)
ax.set_title('FERA NKT — Model Improvement Summary\nRekarisk vs PHAST (Positive = Rekarisk higher)',
             fontweight='bold', fontsize=15)
ax.set_ylim(-80, 60)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_05_improvement_summary.png', dpi=200, bbox_inches='tight')
plt.close()
print('Chart saved')

# Copy to workspace
import shutil
shutil.copy(f'{OUT}/fera_05_improvement_summary.png', '/home/arienugraha-rei/.openclaw/workspace/fera_05_improvement_summary.png')
print('Copied')
