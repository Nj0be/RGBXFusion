"""
plot_combine_analysis.py
Figure dall'analisi del layer `combine` (fuser) del CBAM.
(a) t_ratio per livello FPN, FLIR vs M3FD vs STF (checkpoint Full)
(b) STF: quota Gated vs condizione meteo (gradiente fisico)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(__file__)

# ── (a) t_ratio per livello (Full di ogni dataset), MLP ──────────────────────
levels = [0, 1, 2, 3, 4]
flir = [63.1, 64.9, 65.0, 60.4, 47.2]   # FLIR full  (T=thermal)
m3fd = [45.3, 39.8, 43.0, 53.0, 50.8]   # M3FD full  (T=thermal)
stf  = [31.2, 30.7, 31.8, 30.4, 39.5]   # STF full   (T=gated)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor='white')

ax1.axhline(50, color='#999', ls='--', lw=1)
ax1.plot(levels, flir, '-o', color='#d6604d', lw=2, label='FLIR (T=thermal)')
ax1.plot(levels, m3fd, '-s', color='#2166ac', lw=2, label='M3FD (T=thermal)')
ax1.plot(levels, stf,  '-^', color='#2e8b57', lw=2, label='STF (T=gated)')
ax1.set_xticks(levels)
ax1.set_xlabel('Livello FPN del CBAM', fontsize=10)
ax1.set_ylabel('t_ratio combine  (% modalita "T")', fontsize=10)
ax1.set_title('(a) Preferenza del fuser per livello (checkpoint Full)',
              fontsize=10, fontweight='bold')
ax1.set_ylim(25, 70)
ax1.text(0.05, 66, 'sopra 50% = T (thermal/gated) domina', fontsize=8, color='#555')
ax1.legend(fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)

# ── (b) STF: quota Gated vs condizione (media lvl 0-3) ───────────────────────
conds = ['clear_day', 'clear_night', 'snow_day', 'snow_night', 'fog_night', 'fog_day']
gated = [34.0, 35.7, 36.7, 38.1, 39.2, 44.8]
colors = ['#9ecae1', '#6baed6', '#74c476', '#41ab5d', '#fdae6b', '#e6550d']

bars = ax2.bar(range(len(conds)), gated, color=colors)
ax2.axhline(50, color='#999', ls='--', lw=1)
ax2.set_xticks(range(len(conds)))
ax2.set_xticklabels(conds, rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('quota Gated nel combine (%)', fontsize=10)
ax2.set_title('(b) STF: il Gated cresce col peggiorare del meteo',
              fontsize=10, fontweight='bold')
ax2.set_ylim(0, 55)
for r, v in zip(bars, gated):
    ax2.text(r.get_x()+r.get_width()/2, v+0.8, f'{v:.0f}', ha='center', fontsize=8)
ax2.annotate('clear < snow < fog', xy=(0.5, 0.92), xycoords='axes fraction',
             fontsize=9, color='#e6550d', fontweight='bold', ha='center')
ax2.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
p = os.path.join(OUT, 'combine_analysis.png')
fig.savefig(p, dpi=150, bbox_inches='tight')
print('saved:', p)
