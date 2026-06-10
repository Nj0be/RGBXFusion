#!/usr/bin/env python
"""
run_ablation_m3fd.py
====================
Launcher dell'ablation di modalita' per il dataset M3FD.

NON duplica la logica: richiama validate_fusion_ablation.py (lo stesso core usato
per FLIR) riempiendo i parametri specifici di M3FD, che sono facili da sbagliare:
  - num_classes = 6
  - mean/std RGB e Thermal di M3FD (valori dal README)
  - mapping scena -> (nome dataset, cartella checkpoint)

Esempi:
  # una scena
  python run_ablation_m3fd.py --scene night --ablate-mode mean --cbam-backend mlp
  # tutte le scene di M3FD, modalita' mean
  python run_ablation_m3fd.py --scene all --ablate-mode mean --cbam-backend mlp
  # su GPU (Kaggle/Colab)
  python run_ablation_m3fd.py --scene all --ablate-mode mean --device cuda
"""

import argparse
import subprocess
import sys
import os

# ── Costanti M3FD (dal README) ───────────────────────────────────────────────
M3FD_NUM_CLASSES = 6
M3FD_RGB_MEAN     = ['0.49151019', '0.50717567', '0.50293698']
M3FD_RGB_STD      = ['0.1623529',  '0.14178433', '0.13799928']
M3FD_THERMAL_MEAN = ['0.33000296', '0.33000296', '0.33000296']
M3FD_THERMAL_STD  = ['0.18958051', '0.18958051', '0.18958051']

# scena -> (nome dataset nel factory, cartella checkpoint sotto Fusion_Models)
SCENES = {
    'day':       ('m3fd_day',       'Day'),
    'night':     ('m3fd_night',     'Night'),
    'overcast':  ('m3fd_overcast',  'Overcast'),
    'challenge': ('m3fd_challenge', 'Challenge'),
    'full':      ('m3fd_full',      'Full'),
}


def build_command(args, scene):
    dataset_name, ckpt_dir = SCENES[scene]
    checkpoint = (args.checkpoint if args.checkpoint
                  else os.path.join(args.ckpt_root, ckpt_dir, 'model_best.pth.tar'))

    cmd = [
        sys.executable, 'validate_fusion_ablation.py', args.root,
        '--dataset', dataset_name,
        '--split', args.split,
        '--num-classes', str(M3FD_NUM_CLASSES),
        '--checkpoint', checkpoint,
        '--rgb_mean',     *M3FD_RGB_MEAN,
        '--rgb_std',      *M3FD_RGB_STD,
        '--thermal_mean', *M3FD_THERMAL_MEAN,
        '--thermal_std',  *M3FD_THERMAL_STD,
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
    p = argparse.ArgumentParser(description='Ablation di modalita su M3FD (wrapper)')
    p.add_argument('--scene', default='full',
                   choices=list(SCENES.keys()) + ['all'],
                   help='scena M3FD da valutare ("all" = tutte)')
    p.add_argument('--ablate-mode', default='mean', choices=['zero', 'mean'])
    p.add_argument('--cbam-backend', default='mlp',
                   choices=['mlp', 'FastKAN', 'EfficientKAN', 'WavKAN', 'EfficientKAN_dual'])
    p.add_argument('--cbam-num-grids', type=int, default=5)
    p.add_argument('--cbam-reduction', type=int, default=16)
    p.add_argument('--root', default='Datasets/M3FD', help='root del dataset M3FD')
    p.add_argument('--ckpt-root', default='Checkpoints/M3FD/Fusion_Models',
                   help='cartella con le sottocartelle Day/Night/Overcast/Challenge/Full')
    p.add_argument('--checkpoint', default=None,
                   help='override del checkpoint (ignora il mapping scena->cartella)')
    p.add_argument('--split', default='test')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    scenes = list(SCENES.keys()) if args.scene == 'all' else [args.scene]

    for scene in scenes:
        cmd, checkpoint = build_command(args, scene)
        print('\n' + '#' * 78)
        print(f'# M3FD ablation | scena={scene} | mode={args.ablate_mode} | '
              f'backend={args.cbam_backend}')
        print(f'# checkpoint: {checkpoint}')
        print('#' * 78)
        if not os.path.isfile(checkpoint):
            print(f'  ATTENZIONE: checkpoint non trovato, scena saltata: {checkpoint}')
            continue
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
