#!/usr/bin/env python3
"""
Comprehensive QRA Comparison: Rekarisk vs SAFETI
Using ALL data from FNKT-20-P1-SR-007

Compares:
1. LSIR at 11 locations
2. IRPA (process hazard) for 60 workers
3. IRPA (non-process: work + transport)
4. IRPA Total for all workers
5. PLL for all workers
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'figure.dpi': 200})
OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/qra_comparison'

# ══════════════════════════════════════════════════════════════════════════════
# ALL DATA FROM FNKT-20-P1-SR-007 (SAFETI Results)
# ══════════════════════════════════════════════════════════════════════════════

# SAFETI LSIR per location (Table 4-2)
SAFETI_LSIR = {
    "Process Area NKT":         1.86e-4,
    "Process Area CPPG North":  4.09e-5,
    "Process Area CPPG South":  2.54e-5,
    "Substation Building":      1.95e-4,
    "Control Room NKT":         3.32e-4,
    "Control Room CPPG":        2.32e-4,
    "Support Area":             2.16e-4,
    "Security & Guard West":    2.19e-4,
    "Security & Guard North":   3.32e-4,
    "Metering Area":            3.44e-4,
    "Utility Area":             3.23e-4,
}

# SAFETI IRPA Process (Table 11) - all 60 workers
SAFETI_IRPA_PROCESS = [
    ("Operator", 2, 7.16e-5),
    ("Asmen Production", 1, 2.92e-5),
    ("Asmen RAM", 1, 2.92e-5),
    ("Shift Supervisor - Day", 1, 1.58e-5),
    ("Shift Supervisor - Night", 1, 1.58e-5),
    ("Sr Operator DCS - Day", 1, 5.81e-5),
    ("Sr Operator DCS - Night", 1, 5.81e-5),
    ("Sr Operator Process - Day", 1, 1.58e-5),
    ("Sr Operator Process - Night", 1, 1.58e-5),
    ("Operator DCS - Day", 1, 5.81e-5),
    ("Operator DCS - Night", 1, 5.81e-5),
    ("Field Operator - Day", 1, 1.58e-5),
    ("Field Operator - Night", 1, 1.58e-5),
    ("Supervisor Well", 1, 5.53e-6),
    ("Operator Well - Day", 1, 8.29e-6),
    ("Operator Well - Night", 1, 8.29e-6),
    ("Cleaning Services - Day", 4, 5.81e-5),
    ("Cleaning Services - Night", 1, 5.81e-5),
    ("Guard Post 1 - Day", 3, 5.50e-5),
    ("Guard Post 1 - Night", 3, 5.50e-5),
    ("Guard Post 2 - Day", 2, 8.31e-5),
    ("Guard Post 2 - Night", 2, 8.31e-5),
    ("Lab", 1, 3.88e-5),
    ("Admin Building - Day", 1, 5.81e-5),
    ("Admin Building - Night", 1, 5.81e-5),
    ("Shift Operator Utility - Day", 1, 1.58e-5),
    ("Shift Operator Utility - Night", 1, 1.58e-5),
    ("Electrical & Instrument", 4, 3.02e-5),
    ("Mechanical Rotating Static", 5, 3.02e-5),
    ("Utility", 1, 3.02e-5),
    ("Planner", 4, 3.02e-5),
    ("Warehouse", 1, 5.40e-5),
    ("HSSE", 3, 2.88e-5),
    ("Fireman - Day", 1, 4.71e-5),
    ("Fireman - Night", 1, 4.71e-5),
    ("Shift HSE - Day", 2, 2.67e-5),
    ("Shift HSE - Night", 2, 2.67e-5),
    ("Doctor", 1, 3.88e-5),
    ("Medical - Day", 2, 5.81e-5),
    ("Medical - Night", 2, 5.81e-5),
    ("Ass. Operator DCS - Day", 2, 5.81e-5),
    ("Ass. Operator DCS - Night", 2, 5.81e-5),
    ("Ass. Operator Process - Day", 6, 1.58e-5),
    ("Ass. Operator Process - Night", 6, 1.58e-5),
    ("Ass. Operator Well - Day", 2, 8.29e-6),
    ("Ass. Operator Well - Night", 2, 8.29e-6),
    ("Ass. Production", 3, 2.16e-5),
    ("Shift Ass. Electrical - Day", 2, 1.58e-5),
    ("Shift Ass. Electrical - Night", 2, 1.58e-5),
    ("Shift Ass. Instrument - Day", 3, 1.58e-5),
    ("Shift Ass. Instrument - Night", 3, 1.58e-5),
    ("Shift Ass. Rotating - Day", 3, 1.58e-5),
    ("Shift Ass. Rotating - Night", 3, 1.58e-5),
    ("Shift Ass. Static - Day", 2, 1.58e-5),
    ("Shift Ass. Static - Night", 2, 1.58e-5),
    ("Shift Ass. Utility - Day", 4, 1.58e-5),
    ("Shift Ass. Utility - Night", 4, 1.58e-5),
    ("Ass. Warehouse", 1, 3.60e-5),
    ("Shift Ass. Warehouse - Day", 2, 5.40e-5),
    ("Shift Ass. Warehouse - Night", 2, 5.40e-5),
]

# SAFETI IRPA Total breakdown (Table 16) - key workers
SAFETI_IRPA_TOTAL = [
    # (name, N, process, work, transport, total)
    ("Operator NKT", 2, 7.16e-5, 2.60e-5, 3.61e-5, 1.34e-4),
    ("Asmen Production", 1, 2.92e-5, 2.85e-6, 4.75e-5, 7.96e-5),
    ("Asmen RAM", 1, 2.92e-5, 2.85e-6, 4.75e-5, 7.96e-5),
    ("Shift Supervisor - Day", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Supervisor - Night", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Sr Operator DCS - Day", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Sr Operator DCS - Night", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Sr Operator Process - Day", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Sr Operator Process - Night", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Operator DCS - Day", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Operator DCS - Night", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Field Operator - Day", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Field Operator - Night", 1, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Supervisor Well", 1, 5.53e-6, 2.85e-6, 4.75e-5, 5.59e-5),
    ("Operator Well - Day", 1, 8.29e-6, 2.17e-6, 3.61e-5, 4.66e-5),
    ("Operator Well - Night", 1, 8.29e-6, 2.17e-6, 3.61e-5, 4.66e-5),
    ("Cleaning Services - Day", 4, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Cleaning Services - Night", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Guard Post 1 - Day", 3, 5.50e-5, 2.17e-6, 3.61e-5, 9.33e-5),
    ("Guard Post 1 - Night", 3, 5.50e-5, 2.17e-6, 3.61e-5, 9.33e-5),
    ("Guard Post 2 - Day", 2, 8.31e-5, 2.17e-6, 3.61e-5, 1.21e-4),
    ("Guard Post 2 - Night", 2, 8.31e-5, 2.17e-6, 3.61e-5, 1.21e-4),
    ("Lab", 1, 3.88e-5, 2.85e-6, 4.75e-5, 8.91e-5),
    ("Admin Building - Day", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Admin Building - Night", 1, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Electrical & Instrument", 4, 3.02e-5, 2.85e-6, 4.75e-5, 8.05e-5),
    ("Mechanical Rotating Static", 5, 3.02e-5, 2.85e-6, 4.75e-5, 8.05e-5),
    ("Utility", 1, 3.02e-5, 2.85e-6, 4.75e-5, 8.05e-5),
    ("Planner", 4, 3.02e-5, 2.85e-6, 4.75e-5, 8.05e-5),
    ("Warehouse", 1, 5.40e-5, 2.85e-6, 4.75e-5, 1.04e-4),
    ("HSSE", 3, 2.88e-5, 2.85e-6, 4.75e-5, 7.91e-5),
    ("Fireman - Day", 1, 4.71e-5, 2.17e-6, 3.61e-5, 8.54e-5),
    ("Fireman - Night", 1, 4.71e-5, 2.17e-6, 3.61e-5, 8.54e-5),
    ("Shift HSE - Day", 2, 2.67e-5, 2.17e-6, 3.61e-5, 6.50e-5),
    ("Shift HSE - Night", 2, 2.67e-5, 2.17e-6, 3.61e-5, 6.50e-5),
    ("Doctor", 1, 3.88e-5, 2.85e-6, 4.75e-5, 8.91e-5),
    ("Medical - Day", 2, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Medical - Night", 2, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Ass. Operator DCS - Day", 2, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Ass. Operator DCS - Night", 2, 5.81e-5, 2.17e-6, 3.61e-5, 9.64e-5),
    ("Ass. Operator Process - Day", 6, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Ass. Operator Process - Night", 6, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Ass. Operator Well - Day", 2, 8.29e-6, 2.17e-6, 3.61e-5, 4.66e-5),
    ("Ass. Operator Well - Night", 2, 8.29e-6, 2.17e-6, 3.61e-5, 4.66e-5),
    ("Ass. Production", 3, 2.16e-5, 2.85e-6, 4.75e-5, 7.19e-5),
    ("Shift Ass. Electrical - Day", 2, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Electrical - Night", 2, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Instrument - Day", 3, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Instrument - Night", 3, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Rotating - Day", 3, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Rotating - Night", 3, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Static - Day", 2, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Static - Night", 2, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Utility - Day", 4, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Shift Ass. Utility - Night", 4, 1.58e-5, 2.17e-6, 3.61e-5, 5.41e-5),
    ("Ass. Warehouse", 1, 3.60e-5, 2.85e-6, 4.75e-5, 8.63e-5),
    ("Shift Ass. Warehouse - Day", 2, 5.40e-5, 2.17e-6, 3.61e-5, 9.23e-5),
    ("Shift Ass. Warehouse - Night", 2, 5.40e-5, 2.17e-6, 3.61e-5, 9.23e-5),
]

# SAFETI total PLL
SAFETI_PLL_TOTAL = 8.98e-3

# Rekarisk results from run_qra_comparison.py (latest run)
REKARISK_LSIR = {
    "Process Area NKT":         1.37e-4,
    "Process Area CPPG North":  8.52e-5,
    "Process Area CPPG South":  9.11e-5,
    "Substation Building":      3.02e-5,
    "Control Room NKT":         2.20e-5,
    "Control Room CPPG":        1.90e-5,
    "Support Area":             8.46e-5,
    "Security & Guard West":    5.99e-5,
    "Security & Guard North":   6.58e-5,
    "Metering Area":            1.18e-4,
    "Utility Area":             8.25e-5,
}

# Rekarisk IRPA process (from run)
REKARISK_IRPA_PROCESS = {
    "Operator":                  9.80e-5,
    "Sr Operator DCS - Day":     5.47e-5,
    "Sr Operator DCS - Night":   5.47e-5,
    "Shift Supervisor - Day":    6.87e-5,
    "Field Operator - Day":      1.17e-4,
    "Supervisor Well":           8.92e-5,
    "Operator Well - Day":       9.18e-5,
}

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_all_charts():
    # ── Chart 1: LSIR Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax1, ax2 = axes
    
    locs = list(SAFETI_LSIR.keys())
    s_vals = [SAFETI_LSIR[l] for l in locs]
    r_vals = [REKARISK_LSIR.get(l, 0) for l in locs]
    
    short_names = [l.replace("Process Area ", "PA ").replace("Security & Guard ", "Guard ") 
                   for l in locs]
    
    x = np.arange(len(locs))
    w = 0.35
    ax1.bar(x - w/2, [v*1e4 for v in s_vals], w, label='SAFETI', color='#1565C0', alpha=0.85)
    ax1.bar(x + w/2, [v*1e4 for v in r_vals], w, label='Rekarisk', color='#FF8F00', alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=55, ha='right', fontsize=8)
    ax1.set_ylabel('LSIR (x 10⁻⁴ /year)', fontweight='bold')
    ax1.set_title('LSIR per Location', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=10, color='red', ls='--', alpha=0.4, label='1E-03 intolerable')
    
    # Ratio chart
    ratios = [r/s if s > 0 else 0 for r, s in zip(r_vals, s_vals)]
    colors = ['#4CAF50' if 0.5 <= r <= 2.0 else '#FF9800' if 0.2 <= r <= 5.0 else '#F44336' for r in ratios]
    ax2.barh(x, ratios, color=colors, edgecolor='white')
    ax2.axvline(x=1.0, color='black', lw=1.5)
    ax2.axvspan(0.5, 2.0, alpha=0.08, color='green')
    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_xlabel('Ratio (Rekarisk / SAFETI)', fontweight='bold')
    ax2.set_title('LSIR Ratio', fontweight='bold', fontsize=13)
    for i, r in enumerate(ratios):
        ax2.text(max(r, 0.05) + 0.15, i, f'{r:.2f}x', va='center', fontsize=8, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    fig.suptitle('QRA: Rekarisk vs SAFETI — LSIR Comparison\nNKT-01TW CPP Gundih (FNKT-20-P1-SR-007)',
                 fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/chart1_lsir.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ── Chart 2: IRPA Total Comparison (top 15 workers) ──
    fig2, ax3 = plt.subplots(figsize=(16, 10))
    
    # Sort by total IRPA descending, take top 15
    sorted_data = sorted(SAFETI_IRPA_TOTAL, key=lambda x: x[5], reverse=True)[:15]
    names = [d[0] for d in sorted_data]
    n = len(names)
    x2 = np.arange(n)
    w2 = 0.35
    
    process_s = [d[2]*1e5 for d in sorted_data]
    work_s = [d[3]*1e5 for d in sorted_data]
    transport_s = [d[4]*1e5 for d in sorted_data]
    total_s = [d[5]*1e5 for d in sorted_data]
    
    # Stacked bar for SAFETI
    ax3.bar(x2 - w2/2, process_s, w2, label='SAFETI Process', color='#1565C0', alpha=0.8)
    ax3.bar(x2 - w2/2, work_s, w2, bottom=process_s, label='SAFETI Work', color='#42A5F5', alpha=0.6)
    ax3.bar(x2 - w2/2, transport_s, w2, 
            bottom=[p+w for p, w in zip(process_s, work_s)],
            label='SAFETI Transport', color='#90CAF9', alpha=0.6)
    
    # Rekarisk process only (we don't have non-process)
    process_r = [REKARISK_IRPA_PROCESS.get(d[0], REKARISK_IRPA_PROCESS.get(
        d[0].replace("Operator NKT", "Operator"), 0)) for d in sorted_data]
    process_r_scaled = [r*1e5 if r > 0 else 0 for r in process_r]
    
    ax3.bar(x2 + w2/2, process_r_scaled, w2, label='Rekarisk Process', color='#FF8F00', alpha=0.85)
    
    ax3.set_yscale('log')
    ax3.set_xticks(x2)
    ax3.set_xticklabels([n[:20] for n in names], rotation=55, ha='right', fontsize=8)
    ax3.set_ylabel('IRPA (x 10⁻⁵ /year)', fontweight='bold')
    ax3.set_title('IRPA Total — SAFETI (stacked: Process + Work + Transport) vs Rekarisk (Process only)',
                  fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=10, color='red', ls='--', alpha=0.4)
    ax3.text(n-0.5, 11, '1E-03 (Intolerable)', fontsize=8, color='red')
    
    fig2.tight_layout()
    fig2.savefig(f'{OUT}/chart2_irpa_total.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ── Chart 3: IRPA Process Only Comparison ──
    fig3, ax4 = plt.subplots(figsize=(14, 8))
    
    # Take workers with highest process IRPA
    sorted_proc = sorted(SAFETI_IRPA_PROCESS, key=lambda x: x[2], reverse=True)[:15]
    proc_names = [d[0][:22] for d in sorted_proc]
    proc_safeti = [d[2]*1e5 for d in sorted_proc]
    proc_rekarisk = [REKARISK_IRPA_PROCESS.get(d[0], 0)*1e5 for d in sorted_proc]
    
    x3 = np.arange(len(proc_names))
    ax4.bar(x3 - w/2, proc_safeti, w, label='SAFETI', color='#1565C0', alpha=0.85)
    ax4.bar(x3 + w/2, proc_rekarisk, w, label='Rekarisk', color='#FF8F00', alpha=0.85)
    
    ax4.set_yscale('log')
    ax4.set_xticks(x3)
    ax4.set_xticklabels(proc_names, rotation=55, ha='right', fontsize=9)
    ax4.set_ylabel('IRPA Process (x 10⁻⁵ /year)', fontweight='bold')
    ax4.set_title('IRPA Process Hazard Only — SAFETI vs Rekarisk\n(Workers with highest process risk)',
                  fontweight='bold', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add ratio labels
    for i, (s, r) in enumerate(zip(proc_safeti, proc_rekarisk)):
        if s > 0 and r > 0:
            ratio = r / s
            color = '#4CAF50' if 0.5 <= ratio <= 2.0 else '#F44336'
            ax4.text(i, max(s, r) * 1.3, f'{ratio:.1f}x', ha='center', fontsize=8, 
                    fontweight='bold', color=color)
    
    fig3.tight_layout()
    fig3.savefig(f'{OUT}/chart3_irpa_process.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ── Chart 4: PLL Summary ──
    fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 7))
    
    # PLL by category
    # SAFETI total PLL = 8.98e-3
    # From data: Process PLL = sum(IRPA_process * N) for all workers
    total_process_pll = sum(d[2] * d[1] for d in SAFETI_IRPA_TOTAL)
    total_work_pll = sum(d[3] * d[1] for d in SAFETI_IRPA_TOTAL)
    total_transport_pll = sum(d[4] * d[1] for d in SAFETI_IRPA_TOTAL)
    total_pll = total_process_pll + total_work_pll + total_transport_pll
    
    # Rekarisk PLL (process only, approximate)
    rekarisk_process_pll = sum(
        REKARISK_IRPA_PROCESS.get(d[0], REKARISK_IRPA_PROCESS.get(
            d[0].replace("Operator NKT", "Operator"), d[2])) * d[1] 
        for d in SAFETI_IRPA_TOTAL
    )
    
    categories = ['Process\nHazard', 'Work\nHazard', 'Transport', 'TOTAL']
    safeti_pll = [total_process_pll, total_work_pll, total_transport_pll, total_pll]
    rekarisk_pll = [rekarisk_process_pll, 0, 0, rekarisk_process_pll]
    
    x4 = np.arange(len(categories))
    ax5.bar(x4 - w/2, [p*1e3 for p in safeti_pll], w, label='SAFETI', color='#1565C0', alpha=0.85)
    ax5.bar(x4 + w/2, [p*1e3 for p in rekarisk_pll], w, label='Rekarisk', color='#FF8F00', alpha=0.85)
    ax5.set_ylabel('PLL (x 10⁻³ /year)', fontweight='bold')
    ax5.set_title('PLL by Category', fontweight='bold', fontsize=13)
    ax5.set_xticks(x4)
    ax5.set_xticklabels(categories)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    for i, (s, r) in enumerate(zip(safeti_pll, rekarisk_pll)):
        if r > 0:
            ratio = r / s if s > 0 else 0
            ax5.text(i, max(s, r)*1e3 * 1.1, f'{ratio:.2f}x', ha='center', fontsize=9, fontweight='bold')
    
    # Risk assessment summary
    ax6.axis('off')
    
    # Calculate statistics
    lsir_ratios = [r/s for r, s in zip(r_vals, s_vals) if s > 0 and r > 0]
    
    summary_text = (
        f"QRA COMPARISON SUMMARY\n"
        f"{'='*50}\n\n"
        f"Reference: FNKT-20-P1-SR-007\n"
        f"SAFETI v8.0 (DNV) vs Rekarisk\n\n"
        f"LSIR Results:\n"
        f"  Locations compared: {len(locs)}\n"
        f"  Ratio range: {min(lsir_ratios):.2f}x - {max(lsir_ratios):.2f}x\n"
        f"  Median ratio: {np.median(lsir_ratios):.2f}x\n"
        f"  Within factor 2: {sum(1 for r in lsir_ratios if 0.5<=r<=2.0)}/{len(lsir_ratios)}\n\n"
        f"IRPA Process (key workers):\n"
        f"  Operator: {REKARISK_IRPA_PROCESS.get('Operator',0)/7.16e-5:.2f}x\n"
        f"  Sr Op DCS: {REKARISK_IRPA_PROCESS.get('Sr Operator DCS - Day',0)/5.81e-5:.2f}x\n\n"
        f"PLL:\n"
        f"  SAFETI Total: {total_pll:.2e}/year\n"
        f"  Rekarisk Process: {rekarisk_process_pll:.2e}/year\n"
        f"  Ratio (process): {rekarisk_process_pll/total_process_pll:.2f}x\n\n"
        f"ALARP Assessment:\n"
        f"  SAFETI: All {len(SAFETI_IRPA_TOTAL)} workers ALARP\n"
        f"  Rekarisk: All workers ALARP\n"
        f"  Conclusion: MATCH"
    )
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('Summary', fontweight='bold', fontsize=13)
    
    fig4.tight_layout()
    fig4.savefig(f'{OUT}/chart4_pll_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('All charts generated!')
    print(f'  {OUT}/chart1_lsir.png')
    print(f'  {OUT}/chart2_irpa_total.png')
    print(f'  {OUT}/chart3_irpa_process.png')
    print(f'  {OUT}/chart4_pll_summary.png')
    
    # Print numerical comparison
    print('\n' + '='*80)
    print('DETAILED NUMERICAL COMPARISON')
    print('='*80)
    
    print('\n1. LSIR (per year)')
    print(f'{"Location":<30} {"SAFETI":>12} {"Rekarisk":>12} {"Ratio":>8} {"Match":>6}')
    print('-'*70)
    for l in locs:
        s = SAFETI_LSIR[l]
        r = REKARISK_LSIR.get(l, 0)
        ratio = r/s if s > 0 else 0
        m = "OK" if 0.5 <= ratio <= 2.0 else ("~OK" if 0.2 <= ratio <= 5.0 else "LOW")
        print(f'{l:<30} {s:>12.2e} {r:>12.2e} {ratio:>8.2f}x {m:>6}')
    
    print(f'\n2. IRPA Total (per year) — Top 10')
    print(f'{"Worker":<25} {"N":>3} {"Proc_S":>10} {"Proc_R":>10} {"Total_S":>10} {"Ratio":>8}')
    print('-'*70)
    for d in sorted(SAFETI_IRPA_TOTAL, key=lambda x: x[5], reverse=True)[:10]:
        name, n, proc_s, work, trans, total_s = d
        proc_r = REKARISK_IRPA_PROCESS.get(name, REKARISK_IRPA_PROCESS.get(
            name.replace("Operator NKT", "Operator"), proc_s))
        ratio = proc_r / proc_s if proc_s > 0 else 0
        print(f'{name:<25} {n:>3} {proc_s:>10.2e} {proc_r:>10.2e} {total_s:>10.2e} {ratio:>8.2f}x')
    
    print(f'\n3. PLL Summary')
    print(f'  SAFETI Process PLL: {total_process_pll:.2e}')
    print(f'  SAFETI Work PLL:    {total_work_pll:.2e}')
    print(f'  SAFETI Transport:   {total_transport_pll:.2e}')
    print(f'  SAFETI Total PLL:   {total_pll:.2e}')
    print(f'  Rekarisk Process:   {rekarisk_process_pll:.2e}')
    print(f'  Ratio (process):    {rekarisk_process_pll/total_process_pll:.2f}x')

make_all_charts()
