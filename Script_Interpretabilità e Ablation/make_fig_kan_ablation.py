# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Cali percentuali di mAP rimuovendo una modalita (dual EfficientKAN)
# FLIR scena day, M3FD scena full
groups = ['FLIR', 'M3FD']
calo_thermal = [33.9, 10.6]
calo_rgb     = [13.6, 24.7]

RED = '#C00000'   # togli Thermal
BLUE = '#1F6FB4'  # togli RGB

x = np.arange(len(groups))
w = 0.36

fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
b1 = ax.bar(x - w/2, calo_thermal, w, label='togli Thermal', color=RED)
b2 = ax.bar(x + w/2, calo_rgb,     w, label='togli RGB',     color=BLUE)

for bars in (b1, b2):
    for r in bars:
        ax.annotate(f'{r.get_height():.1f}%',
                    (r.get_x() + r.get_width()/2, r.get_height()),
                    ha='center', va='bottom', fontsize=9,
                    xytext=(0, 2), textcoords='offset points')

ax.set_ylabel('Calo mAP rimuovendo la modalità (%)', fontsize=10)
ax.set_title('Dual EfficientKAN: la dominanza di modalità si inverte', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0, max(calo_thermal + calo_rgb) * 1.25)
ax.legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0, calo_thermal[0]+3.2, 'Termico\ndominante', ha='center', fontsize=8, color=RED)
ax.text(1, calo_rgb[1]+1.0, 'RGB\ndominante', ha='center', fontsize=8, color=BLUE)
plt.tight_layout()

out1 = r'figures/ablation/ablation_crossdataset_kan.png'
plt.savefig(out1, bbox_inches='tight')
print('saved', out1)
