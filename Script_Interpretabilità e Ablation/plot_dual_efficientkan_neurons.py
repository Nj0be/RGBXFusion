#!/usr/bin/env python
"""Plotta alcuni 'neuroni' (funzioni apprese sugli edge) della EfficientKAN_dual:
rami fc_rgb e fc_thermal separati. Per ogni edge:
- SPLINE = b_splines(x) @ scaled_spline_weight  (nonlinearita' KAN)
- BASE   = w * SiLU(x)                           (ramo lineare = MLP)

Layout: righe = (ramo, layer), colonne = edge piu' attivi (norma spline).
"""
import argparse
import os
import re
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.KAN.efficient_kan.kan import KAN as EfficientKAN

SPLINE_ORDER = 3


def load_sub(sd, lvl, branch):
    pre = f'fusion_cbam{lvl}.{branch}.'
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}


def dims_from_sub(sub):
    n_layers = len({int(m) for k in sub for m in re.findall(r'layers\.(\d+)\.', k)})
    dims = []
    for L in range(n_layers):
        bw = sub[f'layers.{L}.base_weight']
        if L == 0:
            dims.append(bw.shape[1])
        dims.append(bw.shape[0])
    gs = sub['layers.0.spline_weight'].shape[-1] - SPLINE_ORDER
    return dims, gs


def edge_curves(layer, n_edges, n_pts=400):
    ssw = layer.scaled_spline_weight.detach()
    norms = ssw.pow(2).sum(-1).sqrt()
    flat = torch.topk(norms.flatten(), min(n_edges, norms.numel())).indices
    edges = [(int(o), int(i)) for o, i in
             (divmod(int(f), layer.in_features) for f in flat)]
    gmin, gmax = float(layer.grid.min()), float(layer.grid.max())
    xs = torch.linspace(gmin, gmax, n_pts)
    out = []
    with torch.no_grad():
        for (o, i) in edges:
            xf = torch.zeros(n_pts, layer.in_features); xf[:, i] = xs
            b = layer.b_splines(xf)[:, i, :]
            spline = b @ ssw[o, i]
            bw = layer.base_weight.detach()[o, i].item()
            base = bw * F.silu(xs)
            out.append((xs.numpy(), spline.numpy(), base.numpy(), (o, i)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',
                   default='output_EfficientKAN_dual_16_2/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--branches', nargs='+', default=['fc_thermal', 'fc_rgb'])
    p.add_argument('--n-edges', type=int, default=4)
    p.add_argument('--out', default='figures/ablation/dual_efficientkan_neurons.png')
    a = p.parse_args()

    sd = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)

    panels = []
    for branch in a.branches:
        sub = load_sub(sd, a.level, branch)
        if 'layers.0.spline_weight' not in sub:
            print(f'  ⚠ {branch} non trovato a lvl{a.level}'); continue
        dims, gs = dims_from_sub(sub)
        kan = EfficientKAN(dims, grid_size=gs, spline_order=SPLINE_ORDER)
        kan.load_state_dict(sub, strict=False); kan.eval()
        for Lidx, layer in enumerate(kan.layers):
            panels.append((branch, Lidx, edge_curves(layer, a.n_edges)))

    if not panels:
        raise SystemExit('Niente da plottare.')

    nrow, ncol = len(panels), a.n_edges
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                             facecolor='white', squeeze=False)
    collapsed = total = 0
    for r, (branch, Lidx, curves) in enumerate(panels):
        for c in range(ncol):
            x, spl, base, (o, i) = curves[c]
            ax = axes[r, c]
            ax.plot(x, spl, color='#d6604d', lw=2, label='spline (KAN)')
            ax.plot(x, base, color='#2166ac', lw=1.5, ls='--', label='base = w·SiLU')
            ax.axhline(0, color='#ccc', lw=.6)
            total += 1
            if abs(spl).max() < 0.1 * (abs(base).max() + 1e-9):
                collapsed += 1
            ax.set_title(f'{branch} L{Lidx} ({i}->{o})', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'{branch}\nlayer{Lidx}', fontsize=9, fontweight='bold')
                if r == 0:
                    ax.legend(fontsize=7)

    fig.suptitle(
        f'EfficientKAN_dual — neuroni (edge) — livello fusion_cbam{a.level}\n'
        f'{a.checkpoint}\n'
        f'spline piatta ≈ 0  =>  edge di fatto MLP  (quasi-MLP: {collapsed}/{total} mostrati)',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches='tight')
    print('saved:', a.out)
    print(f'quasi-MLP edge: {collapsed}/{total} mostrati')


if __name__ == '__main__':
    main()
