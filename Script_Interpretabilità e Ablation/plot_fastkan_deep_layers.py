#!/usr/bin/env python
"""Plotta le funzioni univariate apprese dai layer FastKAN DAL SECONDO IN POI
(layer index >= --start-layer, default 1) dei moduli di fusione.

Per ogni edge mostra:
- componente SPLINE = somma pesata delle RBF (la vera nonlinearita' FastKAN)
- componente BASE   = w * SiLU(x)  (ramo lineare residuo = MLP)
Spline piatta (~0) => quel layer e' di fatto un MLP.

Layout: righe = livelli di fusione (fusion_cbam*), colonne = edge piu' attivi
(per norma dei pesi spline) del layer scelto.
"""
import argparse
import re
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.KAN.fastkan.fastkan import FastKAN


def load_sub(sd, lvl):
    pre = f'fusion_cbam{lvl}.fc.'
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}


def dims_from_sub(sub):
    """Ricostruisce [in, h1, ..., out] e num_grids dai pesi salvati."""
    n_layers = len({int(m) for k in sub for m in re.findall(r'layers\.(\d+)\.', k)})
    ng = sub['layers.0.rbf.grid'].shape[0]
    dims = []
    for L in range(n_layers):
        w = sub[f'layers.{L}.spline_linear.weight']  # (out, in*ng)
        out_dim, in_ng = w.shape
        in_dim = in_ng // ng
        if L == 0:
            dims.append(in_dim)
        dims.append(out_dim)
    return dims, ng


def top_edges_for_layer(layer, ng, n_edges):
    w = layer.spline_linear.weight.detach().view(layer.output_dim, layer.input_dim, ng)
    norms = w.pow(2).sum(-1).sqrt()                       # [out, in]
    flat = torch.topk(norms.flatten(), n_edges).indices
    return [(int(o), int(i)) for o, i in
            (divmod(int(f), layer.input_dim) for f in flat)]


def curves_for_layer(layer, edges):
    out = []
    for (o, i) in edges:
        x, y = layer.plot_curve(i, o)                    # spline (RBF) component
        bw = layer.base_linear.weight.detach()[o, i].item()
        base = bw * F.silu(x)
        out.append((x.numpy(), y.numpy(), base.numpy(), (o, i)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',
                   default='output_FastKAN/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar')
    p.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--start-layer', type=int, default=1,
                   help='primo layer da plottare (1 = secondo layer)')
    p.add_argument('--n-edges', type=int, default=4)
    p.add_argument('--out', default='figures/ablation/fastkan_layer1_functions.png')
    a = p.parse_args()

    sd = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)

    # quali (livello, layer) plottare
    panels = []   # (lvl, layer_idx, curves)
    for lvl in a.levels:
        sub = load_sub(sd, lvl)
        if 'layers.0.spline_linear.weight' not in sub:
            continue
        dims, ng = dims_from_sub(sub)
        kan = FastKAN(dims, num_grids=ng)
        kan.load_state_dict(sub, strict=False)
        kan.eval()
        for Lidx in range(a.start_layer, len(kan.layers)):
            layer = kan.layers[Lidx]
            edges = top_edges_for_layer(layer, ng, a.n_edges)
            panels.append((lvl, Lidx, curves_for_layer(layer, edges)))

    if not panels:
        raise SystemExit('Nessun layer da plottare (controlla --start-layer / checkpoint).')

    nrow = len(panels)
    ncol = a.n_edges
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                             facecolor='white', squeeze=False)
    flat_collapsed = 0
    flat_total = 0
    for r, (lvl, Lidx, curves) in enumerate(panels):
        for c in range(ncol):
            x, spl, base, (o, i) = curves[c]
            ax = axes[r, c]
            ax.plot(x, spl, color='#d6604d', lw=2, label='spline (RBF)')
            ax.plot(x, base, color='#2166ac', lw=1.5, ls='--', label='base = w·SiLU')
            ax.axhline(0, color='#ccc', lw=.6)
            sa = abs(spl).max(); ba = abs(base).max() + 1e-9
            flat_total += 1
            if sa < 0.1 * ba:
                flat_collapsed += 1
            ax.set_title(f'lvl{lvl} L{Lidx}  edge {i}->{o}', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'fusion_cbam{lvl}', fontsize=10, fontweight='bold')
                if r == 0:
                    ax.legend(fontsize=7)

    fig.suptitle(
        f'FastKAN — funzioni apprese sul layer {a.start_layer} (e oltre): spline RBF vs ramo base SiLU\n'
        f'{a.checkpoint}\n'
        f'spline piatta ≈ 0  =>  layer di fatto MLP  '
        f'(quasi-MLP: {flat_collapsed}/{flat_total} edge mostrati)',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches='tight')
    print('saved:', a.out)
    print(f'quasi-MLP edge (spline<10% base): {flat_collapsed}/{flat_total} mostrati')


if __name__ == '__main__':
    main()
