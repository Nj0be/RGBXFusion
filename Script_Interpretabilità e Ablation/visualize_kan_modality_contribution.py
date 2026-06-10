"""
visualize_kan_modality_contribution.py
=======================================
Confronto Thermal vs RGB nei moduli KAN-CBAM su tre checkpoint:
  Day | Night | Both

Tre metriche:
  1. STRUCTURAL — RMS pesi spline per canale [sempre disponibile]
  2. FUNCTIONAL — Sensibilità ai gradienti su dati reali [richiede --data-root]
  3. ACTIVATION — Maschera di channel attention sigmoid(fc(avg)+fc(max)),
                  replica diretta di Fig. 7 del paper RGB-X [richiede --data-root]

Output per ogni cbam-level analizzato:
  {tag}_heatmap.png   — importanza canale×condizione (3 righe: Day/Night/Both)
  {tag}_barplot.png   — T vs RGB aggregato, 3 condizioni affiancate
  {tag}_delta.png     — delta canale-per-canale rispetto a Both
  radar_fpn_*.png     — radar chart quota termica su tutti i livelli FPN
  kan{N}_modality_stats.csv — tabella numerica completa

Usage — solo pesi (nessun dataset):
  python visualize_kan_modality_contribution.py \\
      --checkpoint-day   Checkpoints/.../Day/model_best.pth.tar \\
      --checkpoint-night Checkpoints/.../Night/model_best.pth.tar \\
      --checkpoint-both  Checkpoints/.../Both/model_best.pth.tar \\
      --channels 128 --num-classes 90 \\
      --cbam-levels 0 1 2 3 4

Usage — pesi + gradienti su dati reali:
  python visualize_kan_modality_contribution.py \\
      --checkpoint-day   Checkpoints/.../Day/model_best.pth.tar \\
      --checkpoint-night Checkpoints/.../Night/model_best.pth.tar \\
      --checkpoint-both  Checkpoints/.../Both/model_best.pth.tar \\
      --channels 128 --num-classes 90 \\
      --cbam-levels 0 1 2 3 4 \\
      --data-root /datasets/FLIR_Aligned \\
      --n-samples 256
"""

import argparse
import os
import csv
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.models import load_checkpoint

from models.models import Att_FusionNet


# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS   = ['day', 'night', 'both']
COND_LABELS  = {'day': 'Day', 'night': 'Night', 'both': 'Both'}
COND_COLORS  = {'day': '#f4a82e', 'night': '#4b3a8c', 'both': '#2e8b57'}
COND_HATCHES = {'day': '',        'night': '//',       'both': 'xx'}

THERMAL_COLOR = '#d6604d'
RGB_COLOR     = '#2166ac'


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_args(channels, num_classes, dataset='flir_aligned',
                  att_type='cbam', cbam_backend='FastKAN',
                  cbam_reduction=16, cbam_num_grids=8):
    return types.SimpleNamespace(
        num_classes=num_classes,
        dataset=dataset,
        thermal_checkpoint_path=None,
        rgb_checkpoint_path=None,
        init_fusion_head_weights=None,
        branch='fusion',
        att_type=att_type,
        channels=channels,
        cbam_backend=cbam_backend,
        cbam_reduction=cbam_reduction,
        cbam_num_grids=cbam_num_grids,
    )


def load_model(checkpoint_path, channels, num_classes, att_type, dataset,
               cbam_backend='FastKAN', cbam_reduction=16, cbam_num_grids=8):
    args = _minimal_args(channels, num_classes, dataset, att_type,
                         cbam_backend=cbam_backend,
                         cbam_reduction=cbam_reduction,
                         cbam_num_grids=cbam_num_grids)
    model = Att_FusionNet(args)
    # strict=False: ignora chiavi extra/mancanti (es. fusion_fpn nei ckpt MLP)
    load_checkpoint(model, checkpoint_path, weights_only=False, strict=False)

    # Sanity check: verifica che i pesi del modulo di fusione siano stati
    # EFFETTIVAMENTE caricati. Con strict=False un mismatch di forma (es.
    # cbam_num_grids errato per FastKAN) viene scartato in silenzio lasciando
    # il modulo a init random -> metriche prive di senso. Confrontiamo i pesi
    # del checkpoint con quelli del modello per le chiavi fusion_*.
    import torch as _torch
    _ck = _torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    _sd = _ck.get('state_dict', _ck)
    _msd = model.state_dict()
    _mismatch = []
    for _k, _v in _sd.items():
        if _k.startswith('fusion_') and _k in _msd:
            if tuple(_msd[_k].shape) != tuple(_v.shape):
                _mismatch.append((_k, tuple(_v.shape), tuple(_msd[_k].shape)))
    if _mismatch:
        _lines = '\n'.join(
            f'    {k}: checkpoint={cs} vs model={ms}' for k, cs, ms in _mismatch[:6])
        raise RuntimeError(
            f"[load_model] {len(_mismatch)} pesi di fusione NON caricati per "
            f"mismatch di forma (backend={cbam_backend}, cbam_num_grids="
            f"{cbam_num_grids}). Probabile cbam_num_grids errato.\n{_lines}\n"
            "I pesi sarebbero rimasti a init random. Correggi cbam_num_grids "
            "per combaciare con il checkpoint.")

    model.eval()
    return model


def get_kan_layer(model, cbam_level, kan_layer_idx, att_type):
    cbam = getattr(model, f'fusion_{att_type}{cbam_level}')
    return cbam.fc.layers[kan_layer_idx]


def get_fc_module(model, cbam_level, att_type, cbam_backend, kan_layer_idx=0):
    cbam = getattr(model, f'fusion_{att_type}{cbam_level}')
    if cbam_backend == 'mlp':
        return cbam.fc[0]   # Linear [C/8, 2C]
    else:
        return cbam.fc.layers[kan_layer_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Metrica 1: RMS pesi spline  →  importanza strutturale
# ─────────────────────────────────────────────────────────────────────────────

def weight_rms_per_input(layer, cbam_backend='FastKAN'):
    """
    Importanza strutturale per canale di input: RMS dei pesi che si attaccano
    a quel canale, mediata sui canali di output.

    Tutti i backend ritornano un vettore [in_dim], confrontabile direttamente.
    """
    if cbam_backend == 'mlp':
        # nn.Linear weight: [out_dim, in_dim] -> norma colonne -> [in_dim]
        w = layer.weight.detach()
        return w.pow(2).mean(0).sqrt().cpu().numpy()

    elif cbam_backend == 'FastKAN':
        # FastKANLayer: spline_linear.weight [out_dim, in_dim*num_grids]
        ng   = layer.rbf.num_grids
        w    = layer.spline_linear.weight.detach()
        w_3d = w.view(layer.output_dim, layer.input_dim, ng)
        rms_spline = w_3d.pow(2).mean(-1).sqrt()  # [out, in]
        # Includi anche il base_linear (path residuo SiLU)
        if hasattr(layer, 'base_linear'):
            bw = layer.base_linear.weight.detach()  # [out, in]
            combined = (rms_spline.pow(2) + bw.pow(2)).sqrt()
        else:
            combined = rms_spline
        return combined.mean(0).cpu().numpy()

    elif cbam_backend == 'EfficientKAN':
        # KANLinear (efficient_kan): spline_weight [out, in, grid+order],
        # spline_scaler [out, in] (se enable_standalone_scale_spline),
        # base_weight [out, in]
        sw = layer.spline_weight.detach()
        if (hasattr(layer, 'enable_standalone_scale_spline')
                and layer.enable_standalone_scale_spline
                and hasattr(layer, 'spline_scaler')):
            sw = sw * layer.spline_scaler.unsqueeze(-1).detach()
        rms_spline = sw.pow(2).mean(-1).sqrt()  # [out, in]
        bw = layer.base_weight.detach()         # [out, in]
        combined = (rms_spline.pow(2) + bw.pow(2)).sqrt()
        return combined.mean(0).cpu().numpy()

    elif cbam_backend == 'WavKAN':
        # KANLinear (WavKAN): wavelet_weights [out, in] (peso principale)
        # weight1 [out, in] e' inizializzato ma non usato nel forward attuale
        ww = layer.wavelet_weights.detach()
        return ww.pow(2).mean(0).sqrt().cpu().numpy()

    else:
        raise ValueError(f'Unknown backend: {cbam_backend}')


def _layer_in_out(layer, cbam_backend):
    """Ritorna (in_dim, out_dim, descrittore) compatibile per ciascun backend."""
    if cbam_backend == 'mlp':
        return layer.in_features, layer.out_features, 'MLP Linear'
    elif cbam_backend == 'FastKAN':
        return (layer.input_dim, layer.output_dim,
                f'FastKAN  (grids={layer.rbf.num_grids})')
    elif cbam_backend == 'EfficientKAN':
        return (layer.in_features, layer.out_features,
                f'EfficientKAN  (grid={layer.grid_size}, '
                f'order={layer.spline_order})')
    elif cbam_backend == 'WavKAN':
        return (layer.in_features, layer.out_features,
                f'WavKAN  (wavelet={layer.wavelet_type})')
    else:
        raise ValueError(f'Unknown backend: {cbam_backend}')


# ─────────────────────────────────────────────────────────────────────────────
# Metrica 2: sensibilità ai gradienti  →  importanza funzionale
# ─────────────────────────────────────────────────────────────────────────────

def gradient_sensitivity_per_input(layer, data_loader, n_samples,
                                    device='cpu', cbam_backend='FastKAN'):
    """
    |grad y / grad x| mediato su n_samples campioni reali. Shape -> [in_dim].

    Funziona per tutti i 4 backend tramite firma uniforme layer(x).
    """
    layer = layer.to(device).train()
    in_dim, _, _ = _layer_in_out(layer, cbam_backend)
    grad_accum = torch.zeros(in_dim, device=device)
    count = 0
    for batch in data_loader:
        x = batch[0].to(device).float() if isinstance(batch, (list, tuple)) \
            else batch.to(device).float()
        x.requires_grad_(True)
        if cbam_backend == 'FastKAN':
            y = layer(x, use_layernorm=True)
        else:
            y = layer(x)
        loss = y.norm(dim=1).mean()
        loss.backward()
        grad_accum += x.grad.abs().mean(0).detach()
        x.requires_grad_(False)
        count += 1
        if count * x.shape[0] >= n_samples:
            break
    layer.eval()
    return (grad_accum / max(count, 1)).cpu().numpy()


def dgsm_per_input(layer, data_loader, n_samples,
                   device='cpu', cbam_backend='FastKAN'):
    """
    DGSM (Derivative-based Global Sensitivity Measure):
    E_x[(grad y / grad x)^2] mediato su n_samples campioni reali. Shape -> [in_dim].

    Identica a gradient_sensitivity_per_input ma usa il quadrato del gradiente
    (.pow(2)) invece del valore assoluto (.abs()): penalizza maggiormente le
    sensibilità elevate -> misura di sensibilità globale.

    Funziona per tutti i 4 backend tramite firma uniforme layer(x).
    """
    layer = layer.to(device).train()
    in_dim, _, _ = _layer_in_out(layer, cbam_backend)
    grad_accum = torch.zeros(in_dim, device=device)
    count = 0
    for batch in data_loader:
        x = batch[0].to(device).float() if isinstance(batch, (list, tuple)) \
            else batch.to(device).float()
        x.requires_grad_(True)
        if cbam_backend == 'FastKAN':
            y = layer(x, use_layernorm=True)
        else:
            y = layer(x)
        loss = y.norm(dim=1).mean()
        loss.backward()
        grad_accum += x.grad.pow(2).mean(0).detach()
        x.requires_grad_(False)
        count += 1
        if count * x.shape[0] >= n_samples:
            break
    layer.eval()
    return (grad_accum / max(count, 1)).cpu().numpy()


def _iter_normalized_pairs(data_root, n_samples, batch_size=1, device='cpu',
                           dataset_name='flir_aligned',
                           img_size=(512, 512),
                           rgb_mean=(0.485, 0.456, 0.406),
                           rgb_std=(0.229, 0.224, 0.225),
                           thermal_mean=(0.485, 0.456, 0.406),
                           thermal_std=(0.229, 0.224, 0.225)):
    """
    Generatore di coppie (thermal, rgb) normalizzate dalla pipeline FLIR reale
    (FusionDatasetFLIR + transforms_eval), pronte per il forward del modello.

    Bypassa create_loader (che richiede prefetcher+CUDA): la normalizzazione è
    applicata manualmente su device. Si ferma dopo n_samples campioni.

    Parametri:
        dataset_name: 'flir_aligned' | 'flir_aligned_day' | 'flir_aligned_night'
                      Seleziona il JSON di split corretto da dataset_config.
        img_size:     dimensione immagine come tupla (H, W) o intero.
    """
    import pathlib

    from data import FusionDatasetFLIR
    from data.dataset_config import (
        FlirAlignedFullCfg, FlirAlignedDayCfg, FlirAlignedNightCfg
    )

    # ── Selezione configurazione dataset ─────────────────────────────────────
    _DATASET_CFG = {
        'flir_aligned':       FlirAlignedFullCfg,
        'flir_aligned_day':   FlirAlignedDayCfg,
        'flir_aligned_night': FlirAlignedNightCfg,
    }
    if dataset_name not in _DATASET_CFG:
        raise ValueError(
            f"dataset_name '{dataset_name}' non supportato. "
            f"Scegli tra: {list(_DATASET_CFG.keys())}")

    cfg = _DATASET_CFG[dataset_name]()
    split_cfg = cfg.splits['val']

    data_root = pathlib.Path(data_root)
    thermal_data_dir = data_root / split_cfg['img_dir']
    # RGB: stesso path ma con prefisso rgb (come in FusionDatasetFLIR.__getitem__)
    rgb_data_dir = data_root / split_cfg['img_dir'].replace(
        'images_thermal', 'images_rgb')

    # CocoParser accetta solo CocoParserCfg come argomento
    from effdet.data.parsers.parser_config import CocoParserCfg
    parser_cfg = CocoParserCfg(
        ann_filename=str(data_root / split_cfg['ann_filename']),
        has_labels=split_cfg['has_labels'],
    )
    from effdet.data.parsers.parser_coco import CocoParser
    parser = CocoParser(parser_cfg)

    dataset = FusionDatasetFLIR(
        thermal_data_dir=thermal_data_dir,
        rgb_data_dir=rgb_data_dir,
        parser=parser,
    )

    from data.transforms import transforms_eval

    img_size_tuple = (tuple(img_size) if isinstance(img_size, (list, tuple))
                      else (img_size, img_size))

    dataset.transform = transforms_eval(
        img_size=img_size_tuple,
        interpolation='bilinear',
        use_prefetcher=True,   # produce numpy uint8, gestito da _fusion_collate
        fill_color='mean',
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        thermal_mean=thermal_mean,
        thermal_std=thermal_std,
    )

    # Collate custom: RESIZE esplicito a img_size_tuple con cv2.
    # Necessario perche' transforms_eval di effdet con use_prefetcher=True
    # non garantisce dimensione costante tra immagini, e padding+crop
    # statico lascia comunque variabilita' se le immagini sorgenti sono
    # piu' piccole del target. cv2 resize forza dimensione fissa al 100%.
    target_h, target_w = img_size_tuple
    import cv2 as _cv2  # gia' importato nel modulo, alias locale

    def _fusion_collate(batch):
        # batch: lista di (thermal_np, rgb_np, target)
        # thermal_np/rgb_np: numpy [C, H, W] float32
        import numpy as np
        bs = len(batch)
        th_out  = np.zeros((bs, 3, target_h, target_w), dtype=np.float32)
        rgb_out = np.zeros((bs, 3, target_h, target_w), dtype=np.float32)
        for i, (th, rgb, _) in enumerate(batch):
            # th, rgb: [C=3, H, W] -> trasponi a [H, W, C] per cv2.resize
            th_hwc  = np.transpose(th,  (1, 2, 0))
            rgb_hwc = np.transpose(rgb, (1, 2, 0))
            th_r  = _cv2.resize(th_hwc,  (target_w, target_h),
                                interpolation=_cv2.INTER_LINEAR)
            rgb_r = _cv2.resize(rgb_hwc, (target_w, target_h),
                                interpolation=_cv2.INTER_LINEAR)
            th_out[i]  = np.transpose(th_r,  (2, 0, 1))
            rgb_out[i] = np.transpose(rgb_r, (2, 0, 1))
        return torch.from_numpy(th_out), torch.from_numpy(rgb_out)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_fusion_collate,
    )

    # ── Normalizzazione manuale (float, su device) ────────────────────────────
    # _fusion_collate restituisce tensori float32 [0,255] non normalizzati
    t_mean = torch.tensor([x * 255 for x in thermal_mean], device=device).view(1, 3, 1, 1)
    t_std  = torch.tensor([x * 255 for x in thermal_std],  device=device).view(1, 3, 1, 1)
    r_mean = torch.tensor([x * 255 for x in rgb_mean],     device=device).view(1, 3, 1, 1)
    r_std  = torch.tensor([x * 255 for x in rgb_std],      device=device).view(1, 3, 1, 1)

    collected = 0
    for thermal_raw, rgb_raw in loader:
        thermal = thermal_raw.to(device).sub_(t_mean).div_(t_std)
        rgb     = rgb_raw.to(device).sub_(r_mean).div_(r_std)
        yield thermal, rgb
        collected += thermal.shape[0]
        if collected >= n_samples:
            break


def _build_feature_loader(model, data_root, cbam_level, att_type,
                           n_samples, batch_size=1, device='cpu',
                           dataset_name='flir_aligned',
                           img_size=(512, 512)):
    """
    Raccoglie le feature reali all'ingresso del CBAMLayer al livello cbam_level
    (avg-pool → [B, 2C]), tramite hook sul modulo, per la metrica funzionale
    (gradient_sensitivity_per_input).

    Restituisce un DataLoader di tensori [B, 2*channels].
    """
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    collected_inputs = []

    def _cbam_input_hook(module, inp, out):
        # inp[0]: tensore [B, 2C, H, W] in ingresso al CBAMLayer.forward
        x = inp[0]
        b, c, _, _ = x.shape
        x_avg = torch.nn.functional.adaptive_avg_pool2d(
            x, 1).view(b, c).detach().cpu()
        collected_inputs.append(x_avg)

    cbam   = getattr(model, f'fusion_{att_type}{cbam_level}')
    handle = cbam.register_forward_hook(_cbam_input_hook)

    model = model.to(device).eval()
    with torch.no_grad():
        for thermal, rgb in _iter_normalized_pairs(
                data_root, n_samples, batch_size, device,
                dataset_name, img_size):
            model([thermal, rgb], branch='fusion')

    handle.remove()

    if not collected_inputs:
        raise RuntimeError(
            f"Nessuna feature raccolta per CBAM level {cbam_level}. "
            "Verifica che cbam_level sia nel range corretto e che il "
            "forward pass raggiunga quel livello.")

    all_feats = torch.cat(collected_inputs, dim=0)[:n_samples]
    print(f'  Feature raccolte: {all_feats.shape}  '
          f'(campioni={all_feats.shape[0]}, dim={all_feats.shape[1]})')

    return TorchDataLoader(
        TensorDataset(all_feats),
        batch_size=batch_size,
        shuffle=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrica 3: maschera di channel attention  →  importanza "activation"
#            Replica diretta di Fig. 7 del paper RGB-X (per ogni backend)
# ─────────────────────────────────────────────────────────────────────────────

def attention_mask_per_input(model, data_root, cbam_level, att_type,
                             n_samples, batch_size=1, device='cpu',
                             dataset_name='flir_aligned',
                             img_size=(512, 512)):
    """
    Maschera di channel attention y = sigmoid(fc(avg_pool) + fc(max_pool))
    mediata su n_samples campioni reali. Shape -> [2C].

    È esattamente la grandezza plottata in Fig. 7 del paper RGB-X:
    "normalized attention weights" per canale (thermal nei primi C, RGB nei
    secondi C). Valore alto = il CBAM attende di più su quel canale.

    Indipendente dal backend: replica la chiamata self.fc(...) di _forward_se,
    quindi vale identica per mlp / FastKAN / EfficientKAN / WavKAN.
    """
    cbam = getattr(model, f'fusion_{att_type}{cbam_level}')
    masks = []

    def _mask_hook(module, inp, out):
        # inp[0]: tensore [B, 2C, H, W] in ingresso al CBAMLayer.forward
        x = inp[0]
        b, c, _, _ = x.shape
        x_avg = module.fc(module.avg_pool(x).view(b, c))
        x_max = module.fc(module.max_pool(x).view(b, c))
        y = torch.sigmoid(x_avg + x_max)   # [B, 2C] — maschera di Fig. 7
        masks.append(y.detach().cpu())

    handle = cbam.register_forward_hook(_mask_hook)

    model = model.to(device).eval()
    with torch.no_grad():
        for thermal, rgb in _iter_normalized_pairs(
                data_root, n_samples, batch_size, device,
                dataset_name, img_size):
            model([thermal, rgb], branch='fusion')

    handle.remove()

    if not masks:
        raise RuntimeError(
            f"Nessuna maschera raccolta per CBAM level {cbam_level}.")

    all_masks = torch.cat(masks, dim=0)[:n_samples]
    return all_masks.mean(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Utility split
# ─────────────────────────────────────────────────────────────────────────────

def split_modalities(importance, channels):
    """
    Restituisce (thermal_arr, rgb_arr, t_mean, r_mean, t_ratio)
    t_ratio = T / (T + RGB) ∈ [0,1]  — quota termica
    """
    th     = importance[:channels]
    rgb    = importance[channels:channels * 2]
    t_mean = float(th.mean())
    r_mean = float(rgb.mean())
    t_ratio = t_mean / (t_mean + r_mean + 1e-12)
    return th, rgb, t_mean, r_mean, t_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Heatmap canale × condizione
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmap(imp_dict, channels, metric_label, tag):
    """
    Heatmap [3 righe × in_dim colonne]: Day / Night / Both.
    """
    data   = np.stack([imp_dict[c] for c in CONDITIONS])  # [n_cond, in_dim]
    in_dim = data.shape[1]

    width  = max(10, min(20, in_dim * 0.09 + 3))
    fig, ax = plt.subplots(figsize=(width, max(2.8, 0.55 * len(CONDITIONS) + 1.2)),
                           facecolor='white')

    vmax = np.percentile(data, 98)
    im   = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=0, vmax=vmax)

    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_xlabel('Input channel index', fontsize=9)
    ax.set_title(f'{metric_label}  —  channel importance  [{tag}]',
                 fontsize=10, fontweight='bold')

    if in_dim >= 2 * channels:
        ax.axvline(channels - 0.5, color='white', linewidth=2, linestyle='--')
        y_ann = -0.7
        ax.text(channels / 2,         y_ann, 'Thermal', ha='center',
                fontsize=8, color=THERMAL_COLOR, fontweight='bold',
                transform=ax.transData)
        ax.text(channels + channels/2, y_ann, 'RGB',    ha='center',
                fontsize=8, color=RGB_COLOR,     fontweight='bold',
                transform=ax.transData)

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label='Importance')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Barplot aggregato T vs RGB — 3 condizioni affiancate
# ─────────────────────────────────────────────────────────────────────────────

def fig_barplot(imp_dict, channels, metric_label, tag):
    """
    Gruppi: [Thermal | RGB], barre: Day / Night / Both.
    Annota t_ratio % sopra la barra Thermal di ogni condizione.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')

    n_cond = len(CONDITIONS)
    width = min(0.22, 0.66 / max(1, n_cond))
    x     = np.array([0.0, max(0.75, width * n_cond + 0.3)])  # Thermal | RGB

    stats = {c: split_modalities(imp_dict[c], channels) for c in CONDITIONS
             if c in imp_dict}

    ymax = 0.0
    for i, cond in enumerate(CONDITIONS):
        if cond not in stats:
            continue
        _, _, t_mean, r_mean, t_ratio = stats[cond]
        offset = (i - (n_cond - 1) / 2) * width
        bars = ax.bar(x + offset, [t_mean, r_mean],
                      width=width * 0.9,
                      color=[THERMAL_COLOR, RGB_COLOR],
                      alpha=0.88,
                      label=COND_LABELS[cond],
                      edgecolor=COND_COLORS[cond],
                      linewidth=1.6,
                      hatch=COND_HATCHES[cond])
        ymax = max(ymax, t_mean, r_mean)
        # Annota t_ratio sulla barra Thermal
        ax.text(x[0] + offset, t_mean + ymax * 0.015,
                f'{t_ratio*100:.0f}%T',
                ha='center', va='bottom', fontsize=7,
                color=COND_COLORS[cond], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Thermal', 'RGB'], fontsize=11)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(f'Mean importance per modality  [{tag}]',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Delta canale-per-canale rispetto a Both
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta(imp_dict, channels, metric_label, tag, ref_condition='both'):
    """
    (condizione − riferimento) per canale, per ogni condizione != riferimento.
    Positivo = quel canale conta di più rispetto al modello di riferimento.
    """
    imp_ref = imp_dict[ref_condition]
    in_dim  = len(imp_ref)
    conds   = [c for c in imp_dict if c != ref_condition]

    fig, axes = plt.subplots(max(1, len(conds)), 1,
                             figsize=(max(10, min(20, in_dim * 0.09 + 3)),
                                      2.5 * max(1, len(conds))),
                             sharex=True, facecolor='white', squeeze=False)
    axes = axes[:, 0]

    for ax, cond in zip(axes, conds):
        delta  = imp_dict[cond] - imp_ref
        colors = np.where(delta >= 0, COND_COLORS[cond], '#cccccc')
        ax.bar(range(in_dim), delta, color=colors, width=1.0, linewidth=0)
        ax.axhline(0, color='black', linewidth=0.7)

        if in_dim >= 2 * channels:
            ax.axvline(channels - 0.5, color='black', linewidth=1,
                       linestyle='--', alpha=0.4)
            ylim = ax.get_ylim()
            ax.text(channels / 2,         ylim[0] * 0.85,
                    'Thermal', ha='center', fontsize=7, color=THERMAL_COLOR)
            ax.text(channels + channels/2, ylim[0] * 0.85,
                    'RGB',     ha='center', fontsize=7, color=RGB_COLOR)

        ax.set_ylabel(f'Δ {COND_LABELS[cond]} − {COND_LABELS[ref_condition]}',
                      fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('Input channel index', fontsize=9)
    fig.suptitle(f'Per-channel delta vs {COND_LABELS[ref_condition]}  '
                 f'[{tag}]  —  {metric_label}',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Radar chart quota termica su livelli FPN multipli
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar_fpn(radar_ratios, metric_label):
    """
    radar_ratios: {cbam_level: {condition: t_ratio}}
    Un asse per livello FPN, un poligono per condizione.
    """
    levels = sorted(radar_ratios.keys())
    n_ax   = len(levels)
    if n_ax < 3:
        return None

    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5),
                           subplot_kw=dict(polar=True),
                           facecolor='white')

    for cond in CONDITIONS:
        vals = [radar_ratios[lv].get(cond, 0.0) for lv in levels]
        vals += vals[:1]
        ax.plot(angles, vals, '-o',
                color=COND_COLORS[cond], linewidth=2, markersize=5,
                label=COND_LABELS[cond])
        ax.fill(angles, vals, alpha=0.08, color=COND_COLORS[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'CBAM\nlevel {lv}' for lv in levels], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25%', '50%', '75%'], fontsize=7,
                       color='#888888')
    ax.axhline(y=0.5, color='#cccccc', linewidth=0.8, linestyle='--')

    ax.set_title(f'Thermal quota (t_ratio) per FPN level\n{metric_label}',
                 fontsize=9, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_metrics, channels, out_path):
    """
    all_metrics: {metric: {cbam_level: {condition: importance_array}}}
    """
    rows = []
    for metric, levels_dict in all_metrics.items():
        for cbam_level, cond_dict in levels_dict.items():
            for cond, imp in cond_dict.items():
                _, _, t_mean, r_mean, t_ratio = split_modalities(imp, channels)
                for ch_idx, val in enumerate(imp):
                    rows.append({
                        'metric':     metric,
                        'cbam_level': cbam_level,
                        'condition':  cond,
                        'channel':    ch_idx,
                        'modality':   'thermal' if ch_idx < channels else 'rgb',
                        'importance': f'{val:.6f}',
                        't_mean':     f'{t_mean:.6f}',
                        'r_mean':     f'{r_mean:.6f}',
                        't_ratio':    f'{t_ratio:.4f}',
                    })
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  CSV → {out_path}')


def save_fig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Thermal vs RGB in KAN-CBAM: Day | Night | Both')
    p.add_argument('--checkpoint-day',   default=None)
    p.add_argument('--checkpoint-night', default=None)
    p.add_argument('--checkpoint-both',  default=None)
    p.add_argument('--checkpoints', nargs='+', default=None, metavar='NAME=PATH',
                   help='Condizioni arbitrarie (es. M3FD): '
                        '"day=.../day.tar night=.../night.tar full=.../full.tar". '
                        'Se presente, sostituisce --checkpoint-day/-night/-both.')
    p.add_argument('--ref-condition', type=str, default='both',
                   help='Condizione di riferimento per i delta (default: both; '
                        'per M3FD usa es. full).')
    p.add_argument('--cbam-reduction', type=int, default=16,
                   help='Deve combaciare col checkpoint (es. 64 per *_64_2).')
    p.add_argument('--cbam-num-grids', type=int, default=8,
                   help='Deve combaciare col checkpoint (es. 2 per *_*_2).')
    p.add_argument('--channels',    type=int, default=128)
    p.add_argument('--num-classes', type=int, default=90)
    p.add_argument('--dataset',     type=str, default='flir_aligned')
    p.add_argument('--att-type',    type=str, default='cbam',
                   choices=['cbam', 'eca', 'shuffle'])
    p.add_argument('--cbam-levels', type=int, nargs='+', default=[0],
                   help='Livelli FPN da analizzare, es. --cbam-levels 0 1 2 3 4')
    p.add_argument('--cbam-backend', type=str, default='FastKAN',
                   choices=['mlp', 'FastKAN', 'EfficientKAN', 'WavKAN'],
                   help='Backend fc nel CBAM (default: FastKAN)')
    p.add_argument('--kan-layer',   type=int, default=0, choices=[0, 1],
                   help='KAN only: 0=compressione (2C->C/8), 1=espansione (C/8->2C)')
    p.add_argument('--data-root',    type=str, default=None,
                   help='Root dataset per gradienti su dati reali (opzionale)')
    p.add_argument('--dataset-name', type=str, default='flir_aligned',
                   choices=['flir_aligned', 'flir_aligned_day', 'flir_aligned_night'],
                   help='Split FLIR per metrica funzionale: usa lo stesso del checkpoint')
    p.add_argument('--img-size',    type=int, nargs=2, default=[768, 768],
                   metavar=('H', 'W'),
                   help='Dimensione immagine per il loader. Deve combaciare con '
                        'config.image_size del modello: 768 768 per '
                        'efficientdetv2_dt (FLIR/M3FD), 1280 1280 per STF')
    p.add_argument('--n-samples',   type=int, default=256)
    p.add_argument('--compute-dgsm', action='store_true',
                   help='Calcola anche la metrica DGSM E[(dy/dx)^2] '
                        '(richiede --data-root; piu lenta, default off)')
    p.add_argument('--device',      type=str, default='cpu')
    p.add_argument('--output-dir',  type=str,
                   default='figures/modality_contribution')
    return p.parse_args()


def _setup_conditions(args):
    """Costruisce ckpt_paths {cond: path} e popola le globali CONDITIONS/
    COND_LABELS/COND_COLORS/COND_HATCHES. Supporta sia il vecchio trio
    day/night/both sia un set arbitrario via --checkpoints nome=path."""
    global CONDITIONS, COND_LABELS, COND_COLORS, COND_HATCHES
    if args.checkpoints:
        ckpt_paths = {}
        for item in args.checkpoints:
            if '=' not in item:
                raise SystemExit(f"--checkpoints: atteso 'nome=path', ricevuto '{item}'")
            name, path = item.split('=', 1)
            ckpt_paths[name.strip()] = path.strip()
        CONDITIONS = list(ckpt_paths.keys())
        COND_LABELS = {c: c.capitalize() for c in CONDITIONS}
        cmap = plt.get_cmap('tab10')
        COND_COLORS = {c: matplotlib.colors.to_hex(cmap(i % 10))
                       for i, c in enumerate(CONDITIONS)}
        hatch_cycle = ['', '//', 'xx', '..', '\\\\', '++']
        COND_HATCHES = {c: hatch_cycle[i % len(hatch_cycle)]
                        for i, c in enumerate(CONDITIONS)}
    else:
        ckpt_paths = {'day':   args.checkpoint_day,
                      'night': args.checkpoint_night,
                      'both':  args.checkpoint_both}
        missing = [c for c, p in ckpt_paths.items() if not p]
        if missing:
            raise SystemExit(
                f"Mancano i checkpoint per: {missing}. Passa --checkpoint-day/"
                "-night/-both oppure --checkpoints nome=path ...")
    return ckpt_paths


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Caricamento ────────────────────────────────────────────────────────
    print('\n[1/4] Caricamento checkpoint...')
    ckpt_paths = _setup_conditions(args)
    models = {}
    for cond, path in ckpt_paths.items():
        print(f'  [{COND_LABELS[cond]:5s}] {path}')
        models[cond] = load_model(path, args.channels, args.num_classes,
                                  args.att_type, args.dataset,
                                  cbam_backend=args.cbam_backend,
                                  cbam_reduction=args.cbam_reduction,
                                  cbam_num_grids=args.cbam_num_grids)
    print('  ✓ Tutti i modelli caricati.')

    # ── 2. Calcolo metriche ───────────────────────────────────────────────────
    # all_metrics[metric][cbam_level][condition] = np.array [in_dim]
    # radar_ratios[metric][cbam_level][condition] = float t_ratio
    all_metrics  = {'weight': {}}
    radar_ratios = {'weight': {}}
    if args.data_root:
        all_metrics['gradient']  = {}
        radar_ratios['gradient'] = {}
        all_metrics['attention']  = {}
        radar_ratios['attention'] = {}
        if args.compute_dgsm:
            all_metrics['dgsm']  = {}
            radar_ratios['dgsm'] = {}

    print(f'\n[2/4] Calcolo metriche — livelli CBAM: {args.cbam_levels}')

    for cbam_level in args.cbam_levels:
        print(f'\n  ── CBAM level {cbam_level} ──')

        layers = {}
        for cond in CONDITIONS:
            try:
                layers[cond] = get_fc_module(
                    models[cond], cbam_level, args.att_type,
                    args.cbam_backend, args.kan_layer)
            except (AttributeError, IndexError) as e:
                print(f'  ⚠ [{cond}] layer non trovato: {e}')

        if not layers:
            continue

        ref = next(iter(layers.values()))
        in_dim, out_dim, descr = _layer_in_out(ref, args.cbam_backend)
        print(f'  Layer ({descr}): {in_dim} -> {out_dim}')

        if in_dim < 2 * args.channels:
            print(f'  WARN: in_dim={in_dim} < 2*channels={2*args.channels}. '
                  'Split T/RGB non applicabile - usa --kan-layer 0.')
            continue

        # Strutturale
        w_imp = {}
        for cond, layer in layers.items():
            w_imp[cond] = weight_rms_per_input(layer, args.cbam_backend)
            _, _, t, r, tr = split_modalities(w_imp[cond], args.channels)
            print(f'  [weight/{cond:5s}]  T={t:.4f}  RGB={r:.4f}  '
                  f't_ratio={tr*100:.1f}%')
        all_metrics['weight'][cbam_level]  = w_imp
        radar_ratios['weight'][cbam_level] = {
            c: split_modalities(w_imp[c], args.channels)[4] for c in w_imp}

        # Funzionale (opzionale)
        if args.data_root and 'gradient' in all_metrics:
            g_imp = {}
            for cond, layer in layers.items():
                try:
                    loader = _build_feature_loader(
                        models[cond], args.data_root, cbam_level,
                        args.att_type, args.n_samples,
                        device=args.device,
                        dataset_name=args.dataset_name,
                        img_size=tuple(args.img_size))
                    g_imp[cond] = gradient_sensitivity_per_input(
                        layer, loader, args.n_samples, args.device,
                        cbam_backend=args.cbam_backend)
                    _, _, t, r, tr = split_modalities(
                        g_imp[cond], args.channels)
                    print(f'  [grad /{cond:5s}]  T={t:.4f}  RGB={r:.4f}  '
                          f't_ratio={tr*100:.1f}%')
                except Exception as e:
                    import traceback
                    print(f'  WARN gradient [{cond}] skipped: {e}')
                    print('  ' + '\n  '.join(
                        traceback.format_exc().splitlines()[-6:]))
            if g_imp:
                all_metrics['gradient'][cbam_level]  = g_imp
                radar_ratios['gradient'][cbam_level] = {
                    c: split_modalities(g_imp[c], args.channels)[4]
                    for c in g_imp}

        # DGSM: derivative-based global sensitivity E[(dy/dx)^2] (opzionale)
        if args.data_root and 'dgsm' in all_metrics:
            d_imp = {}
            for cond, layer in layers.items():
                try:
                    loader = _build_feature_loader(
                        models[cond], args.data_root, cbam_level,
                        args.att_type, args.n_samples,
                        device=args.device,
                        dataset_name=args.dataset_name,
                        img_size=tuple(args.img_size))
                    d_imp[cond] = dgsm_per_input(
                        layer, loader, args.n_samples, args.device,
                        cbam_backend=args.cbam_backend)
                    _, _, t, r, tr = split_modalities(
                        d_imp[cond], args.channels)
                    print(f'  [dgsm /{cond:5s}]  T={t:.4f}  RGB={r:.4f}  '
                          f't_ratio={tr*100:.1f}%')
                except Exception as e:
                    import traceback
                    print(f'  WARN dgsm [{cond}] skipped: {e}')
                    print('  ' + '\n  '.join(
                        traceback.format_exc().splitlines()[-6:]))
            if d_imp:
                all_metrics['dgsm'][cbam_level]  = d_imp
                radar_ratios['dgsm'][cbam_level] = {
                    c: split_modalities(d_imp[c], args.channels)[4]
                    for c in d_imp}

        # Activation: maschera channel attention (replica Fig. 7 del paper)
        if args.data_root and 'attention' in all_metrics:
            a_imp = {}
            for cond in layers:
                try:
                    a_imp[cond] = attention_mask_per_input(
                        models[cond], args.data_root, cbam_level,
                        args.att_type, args.n_samples,
                        device=args.device,
                        dataset_name=args.dataset_name,
                        img_size=tuple(args.img_size))
                    _, _, t, r, tr = split_modalities(
                        a_imp[cond], args.channels)
                    print(f'  [attn /{cond:5s}]  T={t:.4f}  RGB={r:.4f}  '
                          f't_ratio={tr*100:.1f}%')
                except Exception as e:
                    import traceback
                    print(f'  WARN attention [{cond}] skipped: {e}')
                    print('  ' + '\n  '.join(
                        traceback.format_exc().splitlines()[-6:]))
            if a_imp:
                all_metrics['attention'][cbam_level]  = a_imp
                radar_ratios['attention'][cbam_level] = {
                    c: split_modalities(a_imp[c], args.channels)[4]
                    for c in a_imp}

    # ── 3. Figure per livello ─────────────────────────────────────────────────
    print('\n[3/4] Generazione figure...')

    backend_label = args.cbam_backend.upper()
    metric_labels = {
        'weight':    f'Weight importance [{backend_label}] (structural)',
        'gradient':  f'Gradient sensitivity [{backend_label}] (functional)',
        'attention': f'Channel attention [{backend_label}] (Fig.7 activation)',
        'dgsm':      f'DGSM [{backend_label}] (global sensitivity)',
    }

    for metric, levels_dict in all_metrics.items():
        if not levels_dict:
            continue
        mlabel = metric_labels[metric]

        for cbam_level, imp_dict in levels_dict.items():
            tag = f'cbam{cbam_level}_kan{args.kan_layer}_{metric}'
            base = os.path.join(args.output_dir, tag)

            save_fig(fig_heatmap(imp_dict, args.channels, mlabel, tag),
                     f'{base}_heatmap.png')
            save_fig(fig_barplot(imp_dict, args.channels, mlabel, tag),
                     f'{base}_barplot.png')
            if args.ref_condition in imp_dict:
                save_fig(fig_delta(imp_dict, args.channels, mlabel, tag,
                                   ref_condition=args.ref_condition),
                         f'{base}_delta.png')

        # Radar FPN se ≥ 3 livelli
        if len(levels_dict) >= 3:
            f = fig_radar_fpn(radar_ratios[metric], mlabel)
            if f is not None:
                save_fig(f, os.path.join(
                    args.output_dir,
                    f'radar_fpn_kan{args.kan_layer}_{metric}.png'))

    # ── 4. CSV ────────────────────────────────────────────────────────────────
    print('\n[4/4] Export CSV...')
    export_csv(all_metrics, args.channels,
               os.path.join(args.output_dir,
                            f'kan{args.kan_layer}_modality_stats.csv'))

    print('\n✓ Completato.')


if __name__ == '__main__':
    main()