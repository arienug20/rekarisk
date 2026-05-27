#!/usr/bin/env python3
"""Generate final improvement comparison chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'figure.dpi': 200})
OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario'

categories = [
    'Release Rate\n(Gas Fullbore)',
    'Jet Fire\nFlame Length',
    'Jet Fire\n4.73 kW/m²',
    'Jet Fire\n6.3 kW/m²',
    'Jet Fire\n12.5 kW/m²',
    'Jet Fire\n37.5 kW/m²',
    'Liquid Release\n(Small/Medium)',
]

before = [+9.5, +9.5, -9.6, -19.0, -40.3, -72.2, -90.0]
after  = [+9.5, +9.5, +4.4, -2.4, -19.7, -50.6, -12.6]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(categories))
w = 0.35

bars1 = ax.bar(x - w/2, before, w, label='Sebelum (Old)', color='#F44336', edgecolor='white', linewidth=1.5, alpha=0.6)
bars2 = ax.bar(x + w/2, after, w, label='Sesudah (Improved)', color='#4CAF50', edgecolor='white', linewidth=1.5)

for bar in bars2:
    val = bar.get_height()
    y_pos = val + (2 if val >= 0 else -3)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:+.0f}%', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=11, fontweight='bold', color='#2E7D32')

ax.axhline(y=0, color='black', linewidth=1.5)
ax.axhline(y=20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.3)
ax.axhline(y=-20, color='#4CAF50', linewidth=1, linestyle='--', alpha=0.3)

legend_elements = [
    Patch(facecolor='#F44336', alpha=0.6, label='Sebelum (Old Model)'),
    Patch(facecolor='#4CAF50', label='Sesudah (Improved)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold', fontsize=11)
ax.set_ylabel('Deviation from PHAST (%)', fontweight='bold', fontsize=13)
ax.set_title('FERA NKT — Perbaikan Model Rekarisk vs PHAST\n(Positive = Rekarisk lebih tinggi, Negative = lebih rendah)',
             fontweight='bold', fontsize=14)
ax.set_ylim(-80, 30)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fera_06_final_improvement.png', dpi=200, bbox_inches='tight')
plt.close()

import shutil
shutil.copy(f'{OUT}/fera_06_final_improvement.png', '/home/arienugraha-rei/.openclaw/workspace/fera_06_final_improvement.png')
print('Done')
