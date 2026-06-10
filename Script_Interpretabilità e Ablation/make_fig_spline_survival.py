# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RED = '#C00000'   # Thermal
BLUE = '#1F6FB4'  # RGB

# FLIR dual, livello 0 (norma pesi spline, layer0)
flir = {'Thermal': 2.143, 'RGB': 0.000}
# M3FD dual, per livello FPN (norma pesi spline, layer0)
lvls = ['0', '1', '2', '3', '4']
m3fd_thermal = [0.000, 0.000, 0.000, 0.000, 0.000]
m3fd_rgb     = [0.088, 0.049, 0.001, 0.000, 0.000]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.8, 3.2), dpi=200,
                               gridspec_kw={'width_ratios': [1, 1.7]})

# Pannello A: FLIR livello 0
bA = axA.bar(list(flir.keys()), list(flir.values()), color=[RED, BLUE], width=0.6)
for r in bA:
    axA.annotate(f'{r.get_height():.2f}',
                 (r.get_x()+r.get_width()/2, r.get_height()),
                 ha='center', va='bottom', fontsize=9, xytext=(0,2),
                 textcoords='offset points')
axA.set_title('FLIR (livello 0)', fontsize=10)
axA.set_ylabel('norma dei pesi spline', fontsize=10)
axA.set_ylim(0, 2.6)
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)

# Pannello B: M3FD per livello
x = np.arange(len(lvls)); w = 0.38
axB.bar(x - w/2, m3fd_thermal, w, color=RED, label='Thermal')
bR = axB.bar(x + w/2, m3fd_rgb, w, color=BLUE, label='RGB')
for r in bR:
    if r.get_height() > 0.0005:
        axB.annotate(f'{r.get_height():.3f}',
                     (r.get_x()+r.get_width()/2, r.get_height()),
                     ha='center', va='bottom', fontsize=8, xytext=(0,2),
                     textcoords='offset points')
axB.set_title('M3FD (per livello FPN)', fontsize=10)
axB.set_xlabel('livello FPN', fontsize=10)
axB.set_xticks(x); axB.set_xticklabels(lvls)
axB.set_ylim(0, 0.11)
axB.legend(frameon=False, fontsize=9)
axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

fig.suptitle('Sopravvivenza delle spline: segue la modalità dominante e si inverte',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = 'figures/ablation/spline_survival.png'
plt.savefig(out, bbox_inches='tight')
print('saved', out)
