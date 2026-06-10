#!/usr/bin/env python
"""
run_ablation_stf.py
===================
Launcher dell'ablation di modalita' per il dataset STF (RGB + Gated).

NON duplica la logica: richiama validate_fusion_ablation.py (lo stesso core di
FLIR/M3FD) riempiendo i parametri specifici di STF:
  - num_classes = 4
  - NIENTE override di mean/std (STF usa i default ImageNet, come nel README)
  - immagini 1280x1280 (forzato in models.py per 'stf' -> molto lento su CPU)
  - mapping scena -> (nome dataset, cartella checkpoint sotto Fusion_Models)

Note su STF:
  - Il "ramo thermal" dell'architettura ospita la modalita' GATED.
    Quindi nella tabella: no_thermal = rimuovi GATED, no_rgb = rimuovi RGB.
  - L'evaluator per STF e' PascalEvaluator: la metrica e' mAP@0.50 (stile VOC),
    NON COCO mAP@[.5:.95]; la colonna AP@0.50 separata sara' n/a. I delta % restano
    validi (stesso evaluator per tutte le condizioni). Lo script di ablation
    etichetta gia' la metrica come "mAP@0.50(VOC)" in questo caso.

Esempi:
  python run_ablation_stf.py --scene fog_night --ablate-mode mean
  python run_ablation_stf.py --scene all --ablate-mode mean --device cuda
"""

import argparse
import subprocess
import sys
import os

STF_NUM_CLASSES = 4

# scena -> (nome dataset nel factory, cartella checkpoint sotto <ckpt_root>)
SCENES = {
    'clear_day':   ('stf_clear_day',   'Clear_Day'),
    'clear_night': ('stf_clear_night', 'Clear_Night'),
    'fog_day':     ('stf_fog_day',     'Fog_Day'),
    'fog_night':   ('stf_fog_night',   'Fog_Night'),
    'snow_day':    ('stf_snow_day',    'Snow_Day'),
    'snow_night':  ('stf_snow_night',  'Snow_Night'),
    'full':        ('stf_full',        'Full'),
}


def build_command(args, scene):
    dataset_name, ckpt_dir = SCENES[scene]
    checkpoint = (args.checkpoint if args.checkpoint
                  else os.path.join(args.ckpt_root, ckpt_dir, 'model_best.pth.tar'))

    cmd = [
        sys.executable, 'validate_fusion_ablation.py', args.root,
        '--dataset', dataset_name,
        '--split', args.split,
        '--num-classes', str(STF_NUM_CLASSES),
        '--checkpoint', checkpoint,
        # NIENTE mean/std: STF usa i default ImageNet (come nel README)
        '--model', 'efficientdetv2_dt',
        '--batch-size', str(args.batch_size),
        '--att_type', 'cbam',
        '--cbam-backend', args.cbam_backend,
        '--cbam-num-grids', str(args.cbam_num_grids),
        '--cbam-reduction', str(args.cbam_reduction),
        '--ablate-mode', args.ablate_mode,
        '--device', args.device,
        '-j', str(args.workers),
    ]
    return cmd, checkpoint


def main():
    p = argparse.ArgumentParser(description='Ablation di modalita su STF (wrapper)')
    p.add_argument('--scene', default='full',
                   choices=list(SCENES.keys()) + ['all'],
                   help='scena STF da valutare ("all" = tutte)')
    p.add_argument('--ablate-mode', default='mean', choices=['zero', 'mean'])
    p.add_argument('--cbam-backend', default='mlp',
                   choices=['mlp', 'FastKAN', 'EfficientKAN', 'WavKAN', 'EfficientKAN_dual'])
    p.add_argument('--cbam-num-grids', type=int, default=5)
    p.add_argument('--cbam-reduction', type=int, default=16)
    p.add_argument('--root', default='Datasets/STF', help='root del dataset STF')
    p.add_argument('--ckpt-root', default='Checkpoints/STF/Fusion_Models/All_Trained',
                   help='cartella con le sottocartelle Clear_Day/.../Fog_Night/.../Full')
    p.add_argument('--checkpoint', default=None,
                   help='override del checkpoint (ignora il mapping scena->cartella)')
    p.add_argument('--split', default='test')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    scenes = list(SCENES.keys()) if args.scene == 'all' else [args.scene]

    for scene in scenes:
        cmd, checkpoint = build_command(args, scene)
        print('\n' + '#' * 78)
        print(f'# STF ablation | scena={scene} | mode={args.ablate_mode} | '
              f'backend={args.cbam_backend}')
        print(f'# checkpoint: {checkpoint}')
        print('# (no_thermal = rimuovi GATED ; no_rgb = rimuovi RGB)')
        print('#' * 78)
        if not os.path.isfile(checkpoint):
            print(f'  ATTENZIONE: checkpoint non trovato, scena saltata: {checkpoint}')
            continue
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
