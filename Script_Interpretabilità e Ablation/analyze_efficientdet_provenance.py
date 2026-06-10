#!/usr/bin/env python
"""
analyze_efficientdet_provenance.py
==================================
Analisi "di provenienza" dei pesi di un modello di fusione RGB-X.

Risponde alla domanda: quanto delle performance e' EfficientDet ereditato
(congelato) vs realmente addestrato in fusione, e da DOVE viene ogni componente?

Per ogni dataset (FLIR / M3FD / STF) confronta i tensori del checkpoint di
fusione con i backbone monomodali (torch.allclose) e riporta, per ogni
componente (backbone / fpn / class_net / box_net di ciascun ramo + teste di
fusione + CBAM):
  - numero di parametri e % sul totale
  - se e' addestrato in fusione (nome contiene 'fusion_cbam') o congelato
  - la PROVENIENZA: con quale sorgente coincide (termico-finetuned, rgb-finetuned,
    oppure "nessun ckpt di dominio" -> tipicamente COCO).

Tutto su CPU, solo pesi, in pochi secondi. Niente immagini/training necessari.

Uso:
  python analyze_efficientdet_provenance.py
  python analyze_efficientdet_provenance.py --datasets FLIR M3FD
"""

import argparse
import os
import csv
import torch


# ── Config dataset: fusione + backbone monomodali (rgb=None se non esiste) ────
DATASETS = {
    'FLIR': dict(
        fusion='Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar',
        thermal='Checkpoints/FLIR_Aligned/Single_Modality_Models/flir_thermal_backbone.pth.tar',
        rgb=None,  # FLIR: nessun backbone RGB di dominio -> RGB resta COCO
        thermal_name='thermal',
    ),
    'M3FD': dict(
        fusion='Checkpoints/M3FD/Fusion_Models/Full/model_best.pth.tar',
        thermal='Checkpoints/M3FD/Single_Modality_Models/m3fd_thermal_backbone.pth.tar',
        rgb='Checkpoints/M3FD/Single_Modality_Models/m3fd_rgb_backbone.pth.tar',
        thermal_name='thermal',
    ),
    'STF': dict(
        fusion='Checkpoints/STF/Fusion_Models/All_Trained/Full/model_best.pth.tar',
        thermal='Checkpoints/STF/Single_Modality_Models/All_Trained/stf_gated_backbone.pth.tar',
        rgb='Checkpoints/STF/Single_Modality_Models/All_Trained/stf_rgb_backbone.pth.tar',
        thermal_name='gated',
    ),
}

# componenti del modello di fusione (prefisso nel checkpoint di fusione) e
# il corrispondente prefisso nel checkpoint monomodale (EfficientDet puro)
FUSION_COMPONENTS = [
    ('thermal_backbone',  'backbone'),
    ('thermal_fpn',       'fpn'),
    ('thermal_class_net', 'class_net'),
    ('thermal_box_net',   'box_net'),
    ('rgb_backbone',      'backbone'),
    ('rgb_fpn',           'fpn'),
    ('rgb_class_net',     'class_net'),
    ('rgb_box_net',       'box_net'),
    ('fusion_class_net',  'class_net'),
    ('fusion_box_net',    'box_net'),
]


def load_sd(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck.get('state_dict', ck)


def match_fraction(fus_sd, fus_prefix, ref_sd, ref_prefix):
    """% di tensori di fus_prefix che coincidono (allclose) col ref_prefix."""
    if ref_sd is None:
        return None
    n_tot = n_match = 0
    for k, v in fus_sd.items():
        if not k.startswith(fus_prefix + '.'):
            continue
        if not torch.is_floating_point(v):
            continue
        ref_k = ref_prefix + k[len(fus_prefix):]
        rv = ref_sd.get(ref_k)
        if rv is None or rv.shape != v.shape:
            continue
        n_tot += 1
        if torch.allclose(v, rv):
            n_match += 1
    if n_tot == 0:
        return None
    return n_match / n_tot


def count_params(sd, prefix):
    return sum(v.numel() for k, v in sd.items()
               if k.startswith(prefix + '.') and torch.is_floating_point(v))


def analyze(name, cfg):
    print('\n' + '=' * 78)
    print(f' PROVENIENZA PESI — {name}')
    print(f' fusione: {cfg["fusion"]}')
    print('=' * 78)

    fus = load_sd(cfg['fusion'])
    th  = load_sd(cfg['thermal']) if cfg['thermal'] and os.path.isfile(cfg['thermal']) else None
    rgb = load_sd(cfg['rgb']) if cfg['rgb'] and os.path.isfile(cfg['rgb']) else None
    tname = cfg['thermal_name']

    total = sum(v.numel() for v in fus.values() if torch.is_floating_point(v))

    rows = []
    print(f' {"componente":<20}{"params":>12}{"%":>7}  {"stato":<10} provenienza')
    print(' ' + '-' * 76)

    # componenti EfficientDet ereditati
    for comp, ref_pre in FUSION_COMPONENTS:
        p = count_params(fus, comp)
        if p == 0:
            continue
        trained = 'fusion_cbam' in comp  # (nessuno qui; i cbam sono gestiti dopo)
        m_th  = match_fraction(fus, comp, th,  ref_pre)
        m_rgb = match_fraction(fus, comp, rgb, ref_pre)

        # verdetto provenienza
        if m_th == 1.0 and (m_rgb is None or m_rgb < 1.0):
            prov = f'== {tname} (dominio)'
        elif m_rgb == 1.0 and (m_th is None or m_th < 1.0):
            prov = '== rgb (dominio)'
        elif m_th == 1.0 and m_rgb == 1.0:
            prov = f'== {tname} & rgb (identici)'
        elif (m_th is None or m_th == 0.0) and (m_rgb is None or m_rgb == 0.0):
            prov = 'nessun ckpt dominio -> COCO/altro' if comp.startswith('rgb') and rgb is None \
                   else 'diverso da entrambi'
        else:
            parts = []
            if m_th is not None:  parts.append(f'{tname}:{m_th*100:.0f}%')
            if m_rgb is not None: parts.append(f'rgb:{m_rgb*100:.0f}%')
            prov = 'parziale (' + ', '.join(parts) + ')'

        stato = 'ADDESTR.' if trained else 'congelato'
        print(f' {comp:<20}{p:>12,}{100*p/total:>6.1f}%  {stato:<10} {prov}')
        rows.append(dict(dataset=name, component=comp, params=p,
                         pct=f'{100*p/total:.2f}', trained=trained, provenance=prov))

    # CBAM (gli unici addestrati in fusione)
    cbam_keys = [k for k in fus if 'fusion_cbam' in k and torch.is_floating_point(fus[k])]
    cbam_p = sum(fus[k].numel() for k in cbam_keys)
    if cbam_p:
        print(f' {"fusion_cbam (x5)":<20}{cbam_p:>12,}{100*cbam_p/total:>6.1f}%  {"ADDESTR.":<10} addestrato in fusione')
        rows.append(dict(dataset=name, component='fusion_cbam', params=cbam_p,
                         pct=f'{100*cbam_p/total:.2f}', trained=True,
                         provenance='addestrato in fusione'))

    print(' ' + '-' * 76)
    print(f' {"TOTALE":<20}{total:>12,}{100.0:>6.1f}%')
    print(f' Addestrato in fusione: {cbam_p:,} ({100*cbam_p/total:.2f}%)  |  '
          f'Ereditato/congelato: {total-cbam_p:,} ({100*(total-cbam_p)/total:.2f}%)')

    # verdetti chiave
    print('\n VERDETTI:')
    m_rgb_bb = match_fraction(fus, 'rgb_backbone', rgb, 'backbone')
    if rgb is None:
        print(f'  - Ramo RGB: NESSUN backbone di dominio per {name} -> pesi COCO generici.')
    else:
        print(f'  - Ramo RGB backbone == {name} rgb fine-tuned ? '
              f'{"SI" if m_rgb_bb==1.0 else "NO/parziale"}')
    fh_th  = match_fraction(fus, 'fusion_class_net', th,  'class_net')
    fh_rgb = match_fraction(fus, 'fusion_class_net', rgb, 'class_net')
    src = (tname if fh_th == 1.0 else ('rgb' if fh_rgb == 1.0 else 'ignota'))
    print(f'  - Testa di fusione (class_net) inizializzata da: {src} (e congelata).')

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument('--out-csv', default='figures/ablation/efficientdet_provenance.csv')
    args = p.parse_args()

    all_rows = []
    for name in args.datasets:
        cfg = DATASETS[name]
        if not os.path.isfile(cfg['fusion']):
            print(f'\n[{name}] checkpoint di fusione non trovato: {cfg["fusion"]} — salto.')
            continue
        all_rows += analyze(name, cfg)

    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f'\nCSV -> {args.out_csv}')


if __name__ == '__main__':
    main()
