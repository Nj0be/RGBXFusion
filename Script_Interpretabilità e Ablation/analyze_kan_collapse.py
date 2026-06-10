#!/usr/bin/env python
"""
Verifica se una FastKAN addestrata sia degenerata in un MLP: misura il contributo
del percorso SPLINE (vera nonlinearita KAN) vs il percorso BASE lineare residuo
(base_linear(silu(x)), di fatto un MLP) in ogni FastKANLayer dei moduli di fusione.

spline% basso  -> la KAN si appoggia al ramo lineare -> e' di fatto un MLP.
"""
import argparse
import torch
from models.KAN.fastkan.fastkan import FastKAN


def analyze(ckpt, levels, channels=128, reduction=16):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    C2 = 2 * channels
    hidden = C2 // reduction
    print(f"\n=== {ckpt} ===")
    for lvl in levels:
        pre = f'fusion_cbam{lvl}.fc.'
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if not sub:
            print(f"  lvl{lvl}: nessun FastKAN fc (backend non FastKAN?)")
            continue
        ng = sub['layers.0.rbf.grid'].shape[0]
        kan = FastKAN([C2, hidden, C2], num_grids=ng)
        kan.load_state_dict(sub, strict=False)
        kan.eval()
        x = torch.randn(1024, C2)   # post-LayerNorm la scala e' ~unitaria, range griglia ok
        with torch.no_grad():
            for i, layer in enumerate(kan.layers):
                xn = layer.layernorm(x) if layer.layernorm is not None else x
                spline = layer.spline_linear(layer.rbf(xn).view(x.shape[0], -1))
                base = layer.base_linear(layer.base_activation(x))
                s = spline.norm(dim=1).mean().item()
                b = base.norm(dim=1).mean().item()
                frac = 100 * s / (s + b + 1e-12)
                tag = "  <-- quasi MLP" if frac < 25 else ""
                print(f"  lvl{lvl} layer{i} ({layer.input_dim}->{layer.output_dim}): "
                      f"||spline||={s:.3f}  ||base||={b:.3f}  spline%={frac:.1f}%{tag}")
                x = spline + base


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--reduction', type=int, default=16)
    a = p.parse_args()
    analyze(a.checkpoint, a.levels, a.channels, a.reduction)
