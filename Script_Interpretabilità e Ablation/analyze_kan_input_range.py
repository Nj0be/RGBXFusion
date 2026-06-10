#!/usr/bin/env python
"""
Diagnostica del COLLASSO delle spline per gap di range.

Aggancia un forward-hook alle LayerNorm (norm_t / norm_r) della dual EfficientKAN
e raccoglie, sul test set, i DESCRITTORI GREZZI in ingresso, cioe' i descrittori
di canale avg/max-pool che la KAN ORIGINALE (senza normalizzazione) riceverebbe.
Li confronta col supporto della griglia spline [-1.75, 1.75] (grid_range [-1,1],
spline_order 3, num_grids 8). Se la massa cade fuori, le B-spline valgono 0 e il
gradiente sui pesi spline e' nullo per la proprieta' di supporto locale: il ramo
spline non puo' imparare e collassa. Mostra anche la distribuzione DOPO
LayerNorm+tanh (con patch), che rientra nel supporto e spiega il risveglio.

NB: l'FPN e i backbone sono congelati, quindi i descrittori grezzi misurati qui
sono identici a quelli che riceverebbe qualunque backend (MLP/KAN), il che rende
la misura rappresentativa anche del modello senza normalizzazione applicata.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from timm.models import load_checkpoint
from models.models import Att_FusionNet
from models.detector import DetBenchPredictImagePair
from data import create_dataset, create_loader, resolve_input_config
from validate_fusion import parser  # riusa la stessa CLI di validate_fusion

parser.add_argument('--max-batches', type=int, default=40,
                    help='quanti batch del test set processare')
parser.add_argument('--out-fig', type=str, default='/kaggle/working/kan_input_range.png')
parser.add_argument('--support', type=float, default=1.75,
                    help='estremo del supporto effettivo della griglia spline')


def main():
    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    args.pretrained = args.pretrained or not args.checkpoint

    model = Att_FusionNet(args)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=False, weights_only=False)
    bench = DetBenchPredictImagePair(model).cuda().eval()
    cfg = bench.config

    raw_vals, norm_vals = [], []

    def hook(m, inp, out):
        x = inp[0].detach().flatten().float().cpu()
        raw_vals.append(x)
        norm_vals.append(torch.tanh(out.detach()).flatten().float().cpu())

    n_hooks = 0
    for i in range(cfg.num_levels):
        cb = getattr(model, f'fusion_cbam{i}', None)
        if cb is None:
            continue
        for nm in ('norm_t', 'norm_r'):
            if hasattr(cb, nm):
                getattr(cb, nm).register_forward_hook(hook)
                n_hooks += 1
    print('hooks registrati:', n_hooks)
    if n_hooks == 0:
        raise SystemExit('Nessuna LayerNorm trovata: modello non patchato o backend diverso.')

    dataset = create_dataset(args.dataset, args.root, args.split)
    ic = resolve_input_config(args, cfg)
    loader = create_loader(
        dataset, input_size=ic['input_size'], batch_size=args.batch_size,
        use_prefetcher=args.prefetcher, interpolation=ic['interpolation'],
        fill_color=ic['fill_color'], rgb_mean=ic['rgb_mean'], rgb_std=ic['rgb_std'],
        thermal_mean=ic['thermal_mean'], thermal_std=ic['thermal_std'],
        num_workers=args.workers, pin_mem=args.pin_mem)

    with torch.no_grad():
        for i, (t, r, tg) in enumerate(loader):
            bench(t, r, img_info=tg, branch='fusion')
            if i + 1 >= args.max_batches:
                break

    raw = torch.cat(raw_vals).numpy()
    norm = torch.cat(norm_vals).numpy()
    S = args.support
    pct_out = 100.0 * np.mean(np.abs(raw) > S)
    print(f'descrittori grezzi: n={raw.size} min={raw.min():.2f} max={raw.max():.2f} '
          f'mean={raw.mean():.2f} std={raw.std():.2f}')
    print(f'percentuale |grezzo| > {S}: {pct_out:.1f}%')

    rng = np.random.default_rng(0)
    def sub(a, n=200000):
        return a if a.size <= n else a[rng.choice(a.size, n, replace=False)]
    raw_s, norm_s = sub(raw), sub(norm)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=200)
    a1.hist(raw_s, bins=90, color='#7f7f7f')
    a1.axvline(-S, color='#C00000', ls='--'); a1.axvline(S, color='#C00000', ls='--')
    a1.set_title('Descrittori grezzi (ingresso KAN senza patch)', fontsize=10)
    a1.set_xlabel('valore'); a1.set_ylabel('conteggio')
    a1.text(0.03, 0.90, f'{pct_out:.0f}% fuori da [-{S}, {S}]',
            transform=a1.transAxes, color='#C00000', fontsize=10)
    a2.hist(norm_s, bins=90, color='#1F6FB4')
    a2.axvline(-S, color='#C00000', ls='--'); a2.axvline(S, color='#C00000', ls='--')
    a2.set_title('Dopo LayerNorm + tanh (con patch)', fontsize=10)
    a2.set_xlabel('valore')
    for a in (a1, a2):
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle('Perché le spline collassano: input fuori dal supporto della griglia',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out_fig) or '.', exist_ok=True)
    plt.savefig(args.out_fig, bbox_inches='tight')
    print('saved', args.out_fig)


if __name__ == '__main__':
    main()
