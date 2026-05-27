#!/usr/bin/env python3
"""Generate QRA comparison charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'figure.dpi': 200})
OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/qra_comparison'

# Data from latest run
locations = [
    'Process Area\nNKT', 'Process Area\nCPPG North', 'Process Area\nCPPG South',
    'Substation\nBuilding', 'Control Room\nNKT', 'Control Room\nCPPG',
    'Support\nArea', 'Security &\nGuard West', 'Security &\nGuard North',
    'Metering\nArea', 'Utility\nArea'
]
safeti = [1.86e-4, 4.09e-5, 2.54e-5, 1.95e-4, 3.32e-4, 2.32e-4, 2.16e-4, 2.19e-4, 3.32e-4, 3.44e-4, 3.23e-4]
rekarisk = [1.37e-4, 8.52e-5, 9.11e-5, 3.02e-5, 2.20e-5, 1.90e-5, 8.46e-5, 5.99e-5, 6.58e-5, 1.18e-4, 8.25e-5]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Chart 1: LSIR comparison (log scale)
ax1 = axes[0]
x = np.arange(len(locations))
w = 0.35
b1 = ax1.bar(x - w/2, [s*1e4 for s in safeti], w, label='SAFETI (Reference)', color='#1565C0', alpha=0.8)
b2 = ax1.bar(x + w/2, [r*1e4 for r in rekarisk], w, label='Rekarisk', color='#FF8F00', alpha=0.8)

ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels(locations, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('LSIR (x 10⁻⁴ /year)', fontweight='bold')
ax1.set_title('LSIR per Location', fontweight='bold', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add ALARP band
ax1.axhspan(0.01, 10, alpha=0.05, color='green')
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(10.5, 1, '1E-03\n(Intolerable)', fontsize=7, color='red', va='center')

# Chart 2: Ratio chart
ax2 = axes[1]
ratios = [r/s if s > 0 else 0 for r, s in zip(rekarisk, safeti)]
colors = ['#4CAF50' if 0.5 <= r <= 2.0 else '#FF9800' if 0.2 <= r <= 5.0 else '#F44336' for r in ratios]
bars = ax2.barh(x, ratios, color=colors, edgecolor='white', linewidth=1)
ax2.axvline(x=1.0, color='black', linewidth=1.5, linestyle='-')
ax2.axvspan(0.5, 2.0, alpha=0.08, color='green')
ax2.set_yticks(x)
ax2.set_yticklabels(locations, fontsize=9)
ax2.set_xlabel('Ratio (Rekarisk / SAFETI)', fontweight='bold')
ax2.set_title('LSIR Ratio', fontweight='bold', fontsize=14)

for bar, ratio in zip(bars, ratios):
    ax2.text(max(bar.get_width(), 0.05) + 0.1, bar.get_y() + bar.get_height()/2,
             f'{ratio:.2f}x', va='center', fontsize=9, fontweight='bold')

ax2.text(1.8, -1, 'Factor 2x', fontsize=8, color='green', alpha=0.6)
ax2.grid(axis='x', alpha=0.3)

fig.suptitle('QRA Comparison: Rekarisk vs SAFETI\nNKT-01TW CPP Gundih (FNKT-20-P1-SR-007)',
             fontweight='bold', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/qra_lsir_comparison.png', dpi=200, bbox_inches='tight')

# Chart 3: IRPA comparison
fig2, ax3 = plt.subplots(figsize=(12, 6))
workers = ['Operator', 'Sr Op DCS\nDay', 'Sr Op DCS\nNight', 'Shift Sup\nDay', 'Field Op\nDay', 'Supv Well', 'Op Well\nDay']
s_irpa = [7.16e-5, 5.81e-5, 5.81e-5, 1.58e-5, 1.58e-5, 5.53e-6, 8.29e-6]
r_irpa = [9.80e-5, 5.47e-5, 5.47e-5, 6.87e-5, 1.17e-4, 8.92e-5, 9.18e-5]

x2 = np.arange(len(workers))
b3 = ax3.bar(x2 - w/2, [s*1e5 for s in s_irpa], w, label='SAFETI', color='#1565C0', alpha=0.8)
b4 = ax3.bar(x2 + w/2, [r*1e5 for r in r_irpa], w, label='Rekarisk', color='#FF8F00', alpha=0.8)

ax3.set_yscale('log')
ax3.set_xticks(x2)
ax3.set_xticklabels(workers, fontsize=9)
ax3.set_ylabel('IRPA - Process Hazard (x 10⁻⁵ /year)', fontweight='bold')
ax3.set_title('IRPA per Worker Group — Process Hazard Only', fontweight='bold', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

# ALARP band
ax3.axhspan(0.1, 100, alpha=0.05, color='green')
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5)
ax3.text(6.5, 11, '1E-03 (Intolerable)', fontsize=8, color='red')

fig2.tight_layout()
fig2.savefig(f'{OUT}/qra_irpa_comparison.png', dpi=200, bbox_inches='tight')

import shutil
shutil.copy(f'{OUT}/qra_lsir_comparison.png', '/home/arienugraha-rei/.openclaw/workspace/qra_lsir_comparison.png')
shutil.copy(f'{OUT}/qra_irpa_comparison.png', '/home/arienugraha-rei/.openclaw/workspace/qra_irpa_comparison.png')
print('Done!')
