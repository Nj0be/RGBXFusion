#!/usr/bin/env python
"""EfficientKAN: contributo spline (B-spline) vs base lineare (Linear(SiLU(x))).
Input campionati nel range della griglia di ogni layer (le spline sono attive li),
per isolare il collasso dei PESI spline dal problema input-fuori-range."""
import argparse
import torch
import torch.nn.functional as F
from models.KAN.efficient_kan.kan import KAN as EfficientKAN


def analyze(ckpt, levels, channels=128, reduction=16, spline_order=3):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    C2, hidden = 2 * channels, (2 * channels) // reduction
    print(f"\n=== {ckpt} ===")
    for lvl in levels:
        pre = f'fusion_cbam{lvl}.fc.'
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if not sub or 'layers.0.spline_weight' not in sub:
            print(f"  lvl{lvl}: nessun EfficientKAN fc"); continue
        gs = sub['layers.0.spline_weight'].shape[-1] - spline_order
        kan = EfficientKAN([C2, hidden, C2], grid_size=gs, spline_order=spline_order)
        kan.load_state_dict(sub, strict=False)
        kan.eval()
        with torch.no_grad():
            for i, layer in enumerate(kan.layers):
                gmin, gmax = float(layer.grid.min()), float(layer.grid.max())
                x = torch.rand(1024, layer.in_features) * (gmax - gmin) + gmin
                base = F.linear(layer.base_activation(x), layer.base_weight)
                spline = F.linear(layer.b_splines(x).view(x.size(0), -1),
                                  layer.scaled_spline_weight.view(layer.out_features, -1))
                b = base.norm(dim=1).mean().item()
                s = spline.norm(dim=1).mean().item()
                frac = 100 * s / (s + b + 1e-12)
                tag = "  <-- quasi MLP" if frac < 25 else ""
                print(f"  lvl{lvl} layer{i} ({layer.in_features}->{layer.out_features}): "
                      f"spline%={frac:.1f}%  (||spline||={s:.3f} ||base||={b:.3f}){tag}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--reduction', type=int, default=16)
    a = p.parse_args()
    analyze(a.checkpoint, a.levels, a.channels, a.reduction)
