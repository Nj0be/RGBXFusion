"""
plot_ablation_crossdataset.py
Grafico a barre del calo di mAP per modalita' rimossa (ablation mean, MLP),
confronto FLIR (thermal-dominant) vs M3FD (rgb-dominant), per scena.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# calo % di mAP[.5:.95] rimuovendo ciascuna modalita' (ablate-mode mean, MLP)
# struttura: (etichetta, drop_thermal_%, drop_rgb_%)
DATA = [
    ('FLIR\nday',   54.0, 11.2),
    ('FLIR\nnight', 87.9, 13.7),
    ('FLIR\nfull',  61.4,  9.6),
    ('M3FD\nday',   14.1, 34.2),
    ('M3FD\nnight', 16.8, 28.6),
    ('M3FD\nfull',  17.1, 26.9),
]

labels    = [d[0] for d in DATA]
drop_t    = [d[1] for d in DATA]
drop_r    = [d[2] for d in DATA]

x = np.arange(len(labels))
w = 0.38

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
b1 = ax.bar(x - w/2, drop_t, w, label='togli THERMAL (importanza T)',
            color='#d6604d')
b2 = ax.bar(x + w/2, drop_r, w, label='togli RGB (importanza RGB)',
            color='#2166ac')

ax.axvline(2.5, color='#999999', linestyle='--', linewidth=1)
ax.text(1.0, 92, 'FLIR — THERMAL dominante', ha='center', fontsize=9,
        color='#d6604d', fontweight='bold')
ax.text(4.0, 92, 'M3FD — RGB dominante', ha='center', fontsize=9,
        color='#2166ac', fontweight='bold')

ax.set_ylabel('Calo mAP[.5:.95] rimuovendo la modalita  (%)', fontsize=10)
ax.set_title('Ablation causale di modalita (mean, MLP): la dominanza si inverte tra dataset',
             fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 100)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)

for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.2,
                f'{r.get_height():.0f}', ha='center', fontsize=7.5)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'ablation_crossdataset.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved:', out)
