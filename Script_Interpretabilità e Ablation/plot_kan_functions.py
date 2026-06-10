#!/usr/bin/env python
"""
Plotta le funzioni univariate apprese dai layer KAN dei moduli di fusione:
- componente SPLINE (la vera nonlinearita KAN)
- componente BASE = w * SiLU(x) (il ramo lineare residuo = MLP)

Mostra visivamente il collasso: se la spline e' piatta (~0), il layer e' di fatto
un MLP (solo SiLU scalata). Confronta FastKAN vs EfficientKAN, layer 0.
"""
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.KAN.fastkan.fastkan import FastKAN
from models.KAN.efficient_kan.kan import KAN as EfficientKAN


def load_sub(ckpt, lvl):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    pre = f'fusion_cbam{lvl}.fc.'
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}


def fastkan_edges(sub, n_edges, C2=256, hidden=16):
    ng = sub['layers.0.rbf.grid'].shape[0]
    kan = FastKAN([C2, hidden, C2], num_grids=ng); kan.load_state_dict(sub, strict=False); kan.eval()
    layer = kan.layers[0]
    # edge piu' attivi per norma spline
    w = layer.spline_linear.weight.detach().view(layer.output_dim, layer.input_dim, ng)
    norms = w.pow(2).sum(-1).sqrt()                  # [out, in]
    flat = torch.topk(norms.flatten(), n_edges).indices
    edges = [(int(o), int(i)) for o, i in (divmod(int(f), layer.input_dim) for f in flat)]
    curves = []
    for (o, i) in edges:
        x, y = layer.plot_curve(i, o)                # spline component
        bw = layer.base_linear.weight.detach()[o, i].item()
        base = bw * F.silu(x)
        curves.append((x.numpy(), y.numpy(), base.numpy(), (o, i)))
    return curves


def efficientkan_edges(sub, n_edges, C2=256, hidden=16, spline_order=3):
    gs = sub['layers.0.spline_weight'].shape[-1] - spline_order
    kan = EfficientKAN([C2, hidden, C2], grid_size=gs, spline_order=spline_order)
    kan.load_state_dict(sub, strict=False); kan.eval()
    layer = kan.layers[0]
    ssw = layer.scaled_spline_weight.detach()        # [out, in, n_basis]
    norms = ssw.pow(2).sum(-1).sqrt()
    flat = torch.topk(norms.flatten(), n_edges).indices
    edges = [(int(o), int(i)) for o, i in (divmod(int(f), layer.in_features) for f in flat)]
    gmin, gmax = float(layer.grid.min()), float(layer.grid.max())
    xs = torch.linspace(gmin, gmax, 400)
    curves = []
    with torch.no_grad():
        for (o, i) in edges:
            xf = torch.zeros(400, layer.in_features); xf[:, i] = xs
            b = layer.b_splines(xf)[:, i, :]          # [400, n_basis]
            spline = b @ ssw[o, i]                    # [400]
            bw = layer.base_weight.detach()[o, i].item()
            base = bw * F.silu(xs)
            curves.append((xs.numpy(), spline.numpy(), base.numpy(), (o, i)))
    return curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fastkan', default='output_FastKAN/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar')
    p.add_argument('--efficientkan', default='output_EfficientKAN/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar')
    p.add_argument('--n-edges', type=int, default=4)
    p.add_argument('--out', default='figures/ablation/kan_learned_functions.png')
    a = p.parse_args()

    fk = fastkan_edges(load_sub(a.fastkan, 0), a.n_edges)
    ek = efficientkan_edges(load_sub(a.efficientkan, 0), a.n_edges)

    n = a.n_edges
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6), facecolor='white')
    for col in range(n):
        # FastKAN (riga 0)
        x, spl, base, (o, i) = fk[col]
        ax = axes[0, col]
        ax.plot(x, spl, color='#d6604d', lw=2, label='spline (KAN)')
        ax.plot(x, base, color='#2166ac', lw=1.5, ls='--', label='base = w·SiLU')
        ax.axhline(0, color='#ccc', lw=.6)
        ax.set_title(f'FastKAN  edge {i}->{o}', fontsize=9)
        if col == 0:
            ax.set_ylabel('FastKAN', fontsize=11, fontweight='bold'); ax.legend(fontsize=7)
        # EfficientKAN (riga 1)
        x, spl, base, (o, i) = ek[col]
        ax = axes[1, col]
        ax.plot(x, spl, color='#d6604d', lw=2, label='spline (KAN)')
        ax.plot(x, base, color='#2166ac', lw=1.5, ls='--', label='base = w·SiLU')
        ax.axhline(0, color='#ccc', lw=.6)
        ax.set_title(f'EfficientKAN  edge {i}->{o}', fontsize=9)
        if col == 0:
            ax.set_ylabel('EfficientKAN', fontsize=11, fontweight='bold'); ax.legend(fontsize=7)

    fig.suptitle('Funzioni apprese sugli edge del KAN (layer 0): spline vs ramo base SiLU\n'
                 'spline piatta ≈ 0  =>  il layer e\' di fatto un MLP (solo SiLU scalata)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(a.out, dpi=150, bbox_inches='tight')
    print('saved:', a.out)


if __name__ == '__main__':
    main()
