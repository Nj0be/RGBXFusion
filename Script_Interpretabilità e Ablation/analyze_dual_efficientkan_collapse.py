#!/usr/bin/env python
"""EfficientKAN_dual: contributo del ramo spline (B-spline) vs ramo base lineare
(Linear(SiLU(x))) per ENTRAMBI i sotto-rami fc_rgb / fc_thermal.

Per ogni edge KANLinear il forward e':  output = base + spline
    base   = Linear(SiLU(x), base_weight)        # equivalente a un layer MLP
    spline = Linear(b_splines(x), spline_weight)  # la non-linearita' adattiva KAN
Se ||spline|| -> 0 il ramo collassa su un MLP (solo SiLU scalata).

Gli iperparametri (grid_size, hidden/reduction, n. livelli) sono auto-rilevati
dalle forme nel checkpoint, quindi lo script vale per dual_16/32/64/128_*.
Input campionati DENTRO il range della griglia di ogni layer (le spline sono
attive li), per isolare il collasso dei PESI dal problema input-fuori-range.
"""
import argparse
import glob
import os
import re
import csv
import torch
import torch.nn.functional as F
from models.KAN.efficient_kan.kan import KAN as EfficientKAN

SPLINE_ORDER = 3
BRANCHES = ["fc_rgb", "fc_thermal"]


def detect_levels(sd):
    return sorted({int(m) for k in sd for m in re.findall(r"fusion_cbam(\d+)\.", k)})


def build_kan_from_sub(sub):
    """Ricostruisce EfficientKAN([in, hidden, out]) dai pesi salvati."""
    n_layers = len([k for k in sub if k.endswith(".spline_weight")])
    dims = []
    for i in range(n_layers):
        bw = sub[f"layers.{i}.base_weight"]  # (out, in)
        if i == 0:
            dims.append(bw.shape[1])
        dims.append(bw.shape[0])
    gs = sub["layers.0.spline_weight"].shape[-1] - SPLINE_ORDER
    kan = EfficientKAN(dims, grid_size=gs, spline_order=SPLINE_ORDER)
    kan.load_state_dict(sub, strict=False)
    kan.eval()
    return kan, gs


@torch.no_grad()
def analyze_branch(kan, n_samples=2048):
    rows = []
    for i, layer in enumerate(kan.layers):
        gmin, gmax = float(layer.grid.min()), float(layer.grid.max())
        x = torch.rand(n_samples, layer.in_features) * (gmax - gmin) + gmin
        base = F.linear(layer.base_activation(x), layer.base_weight)
        spline = F.linear(
            layer.b_splines(x).view(x.size(0), -1),
            layer.scaled_spline_weight.view(layer.out_features, -1),
        )
        b = base.norm(dim=1).mean().item()
        s = spline.norm(dim=1).mean().item()
        frac = 100 * s / (s + b + 1e-12)
        rows.append((i, layer.in_features, layer.out_features, frac, s, b))
    return rows


def analyze_ckpt(ckpt, writer=None):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    levels = detect_levels(sd)
    folder = re.search(r"(dual_\d+_\d+)", ckpt)
    exp = re.search(r"(DAY|NIGHT|FULL)", ckpt)
    tag_folder = folder.group(1) if folder else "?"
    tag_exp = exp.group(1) if exp else "?"
    print(f"\n=== {ckpt} ===")
    collapsed = total = 0
    for lvl in levels:
        for branch in BRANCHES:
            pre = f"fusion_cbam{lvl}.{branch}."
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if not sub or "layers.0.spline_weight" not in sub:
                continue
            kan, gs = build_kan_from_sub(sub)
            for (i, fin, fout, frac, s, b) in analyze_branch(kan):
                total += 1
                tag = "  <-- quasi MLP" if frac < 25 else ""
                if frac < 25:
                    collapsed += 1
                print(
                    f"  lvl{lvl} {branch:10s} layer{i} ({fin}->{fout}) grid={gs}: "
                    f"spline%={frac:5.1f}%  (||spline||={s:.3f} ||base||={b:.3f}){tag}"
                )
                if writer:
                    writer.writerow([tag_folder, tag_exp, lvl, branch, i,
                                     fin, fout, gs, f"{frac:.2f}", f"{s:.4f}", f"{b:.4f}"])
    if total:
        print(f"  >>> {collapsed}/{total} edge sotto soglia 25% (quasi-MLP)")
    return collapsed, total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="cartella che contiene output_EfficientKAN_dual_*")
    p.add_argument("--ckpt-name", default="model_best.pth.tar")
    p.add_argument("--glob", default="output_EfficientKAN_dual_*_*/train_flir/*/")
    p.add_argument("--csv", default="dual_efficientkan_spline_report.csv")
    a = p.parse_args()

    pattern = os.path.join(a.root, a.glob, a.ckpt_name)
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise SystemExit(f"Nessun checkpoint trovato con: {pattern}")
    print(f"Trovati {len(ckpts)} checkpoint.")

    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "exp", "level", "branch", "layer",
                    "in", "out", "grid", "spline_pct", "norm_spline", "norm_base"])
        tot_c = tot_n = 0
        for ck in ckpts:
            c, n = analyze_ckpt(ck, w)
            tot_c += c
            tot_n += n
    print(f"\n==== TOTALE: {tot_c}/{tot_n} edge quasi-MLP (spline%<25) ====")
    print(f"Report CSV: {a.csv}")
