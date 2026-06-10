#!/usr/bin/env python
"""
analyze_combine_modality.py
===========================
Analisi della preferenza di modalita' nel layer `combine` del CBAM.

Il `combine` (Conv2d 2C->C, 1x1) e' la VERA operazione di fusione di RGBXFusion
(NON presente nel CBAM originale di Woo et al. 2018, che preserva i canali):
riduce l'input concatenato [thermal(C) | rgb(C)] a C canali fusi. I suoi pesi
dicono, in modo strutturale, quanto ciascuna modalita' alimenta l'output fuso.

Per ogni input-channel j calcola la norma L2 della colonna combine.weight[:, j]
(contributo dell'input j a tutti gli output). Aggrega per modalita':
  thermal = canali 0..C-1 , rgb = canali C..2C-1
e riporta la quota termica  t_ratio = T / (T + RGB).

Solo pesi: nessun dato, nessuna GPU, qualsiasi backend (combine e' fuori dal fc).

Uso:
  python analyze_combine_modality.py --checkpoints CK1 [CK2 ...] [--labels L1 L2 ...]
Esempio (FLIR MLP, Day/Night/Full):
  python analyze_combine_modality.py \
    --checkpoints Checkpoints/FLIR_Aligned/Fusion_Models/Day/model_best.pth.tar \
                  Checkpoints/FLIR_Aligned/Fusion_Models/Night/model_best.pth.tar \
                  Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar \
    --labels day night full --channels 128
"""

import argparse
import os
import csv
import torch


def load_sd(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck.get('state_dict', ck)


def combine_modality_per_level(sd, channels, att_type='cbam', num_levels=5):
    """Ritorna lista di dict per livello: {level, T, RGB, t_ratio} dal combine."""
    C = channels
    rows = []
    for i in range(num_levels):
        key = f'fusion_{att_type}{i}.combine.weight'   # [C_out, 2C, 1, 1]
        if key not in sd:
            continue
        w = sd[key].detach().float()
        w = w.view(w.shape[0], w.shape[1])             # [C_out, 2C]
        col_norm = w.pow(2).sum(dim=0).sqrt()          # [2C] norma L2 per input-channel
        th = col_norm[:C]
        rgb = col_norm[C:2 * C]
        t_mean = float(th.mean())
        r_mean = float(rgb.mean())
        t_ratio = t_mean / (t_mean + r_mean + 1e-12)
        rows.append(dict(level=i, T=t_mean, RGB=r_mean, t_ratio=t_ratio))
    return rows


def main():
    p = argparse.ArgumentParser(description='Preferenza di modalita nel combine del CBAM')
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='uno o piu checkpoint di fusione (model_best.pth.tar)')
    p.add_argument('--labels', nargs='+', default=None,
                   help='etichette (es. day night full); default = nome file')
    p.add_argument('--channels', type=int, default=128, help='C per modalita (default 128)')
    p.add_argument('--att-type', default='cbam', choices=['cbam', 'eca', 'shuffle'])
    p.add_argument('--num-levels', type=int, default=5)
    p.add_argument('--out-csv', default='figures/ablation/combine_modality.csv')
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit('--labels deve avere la stessa lunghezza di --checkpoints')

    all_rows = []
    for ckpt, label in zip(args.checkpoints, labels):
        if not os.path.isfile(ckpt):
            print(f'[{label}] checkpoint non trovato: {ckpt} — salto.')
            continue
        sd = load_sd(ckpt)
        rows = combine_modality_per_level(sd, args.channels, args.att_type, args.num_levels)
        if not rows:
            print(f'[{label}] nessun combine trovato (att_type giusto?). Salto.')
            continue

        print('\n' + '=' * 60)
        print(f' COMBINE — preferenza di modalita  [{label}]')
        print(f' {ckpt}')
        print('=' * 60)
        print(f' {"lvl":>3}  {"T":>10}  {"RGB":>10}  {"t_ratio":>8}  dominante')
        print(' ' + '-' * 56)
        for r in rows:
            dom = 'THERMAL' if r['t_ratio'] > 0.5 else 'RGB'
            print(f' {r["level"]:>3}  {r["T"]:>10.4f}  {r["RGB"]:>10.4f}  '
                  f'{r["t_ratio"]*100:>7.1f}%  {dom}')
            all_rows.append(dict(label=label, **r, dominant=dom))
        # media sui livelli "vivi" (escludo lvl 4, tipicamente inattivo)
        live = [r for r in rows if r['level'] != 4]
        if live:
            tr = sum(r['t_ratio'] for r in live) / len(live)
            print(' ' + '-' * 56)
            print(f' media lvl 0-3: t_ratio = {tr*100:.1f}%  '
                  f'({"THERMAL" if tr>0.5 else "RGB"} dominante)')

    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f'\nCSV -> {args.out_csv}')


if __name__ == '__main__':
    main()
