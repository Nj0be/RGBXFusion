#!/usr/bin/env python
"""
validate_fusion_ablation.py
===========================
Ablation di modalita' su un modello di fusione GIA' ADDESTRATO (nessun training).

Misura il contributo CAUSALE di ciascuna modalita' sulla metrica vera (mAP),
spegnendo le feature di un ramo PRIMA della concatenazione nel modulo CBAM
(models.py:91-94) e ri-valutando sullo stesso test set.

Per ogni invocazione lancia 3 condizioni sullo stesso checkpoint/dataset:
    - full      : entrambe le modalita' attive (baseline)
    - no_thermal: ramo thermal spento  -> resta solo RGB
    - no_rgb    : ramo RGB spento       -> resta solo Thermal
e stampa una tabella di confronto con i delta di mAP.

Modalita' di spegnimento (--ablate-mode):
    - zero : feature dell'FPN sostituite con 0. Semplice ma fuori distribuzione
             (dopo norm/sigmoid lo 0 non e' "assenza di segnale"): tende a
             SOVRASTIMARE il calo. Utile come limite superiore.
    - mean : feature sostituite con la media per-canale calcolata sull'intero
             dataset (un primo passaggio). Toglie l'informazione SPECIFICA
             dell'immagine mantenendo la statistica attesa: piu' pulita.
Consigliato riportare entrambe per mostrare la robustezza alla scelta.

L'intervento e' fatto con un forward-hook sull'FPN (thermal_fpn / rgb_fpn):
i pesi del modello NON vengono toccati.

--- Breakdown per scena ---
Non c'e' label di scena nell'evaluator standard: per il confronto day/night/full
si rilancia lo script cambiando --dataset e --checkpoint, esattamente come nel
resto del repo (flir_aligned_day | flir_aligned_night | flir_aligned).

Esempio (FLIR, checkpoint Full, ablation con media):
  python validate_fusion_ablation.py Datasets/FLIR_Aligned \
      --dataset flir_aligned --split test --num-classes 90 \
      --checkpoint Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar \
      --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 \
      --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 \
      --model efficientdetv2_dt --batch-size 8 --branch fusion \
      --att_type cbam --cbam-backend mlp --ablate-mode mean --classwise
"""

import argparse
import time
from contextlib import suppress

import torch
import torch.nn.parallel
import torch.utils.data
from timm.utils import AverageMeter, setup_default_logging
from timm.models import load_checkpoint

from models.models import Att_FusionNet
from models.detector import DetBenchPredictImagePair
from data import create_dataset, resolve_input_config
from data.transforms import transforms_eval
from data.loader import DetectionFastCollate
from utils.evaluator import create_evaluator

torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Loader device-agnostico
# ─────────────────────────────────────────────────────────────────────────────
# Il repo supporta SOLO il prefetcher (transforms_eval ha assert use_prefetcher;
# PrefetchLoader e' interamente su CUDA). Per poter girare anche su CPU si usa
# il transform numpy (use_prefetcher=True -> ImageToNumpy) con un DataLoader
# semplice, e si applica normalizzazione + spostamento su device a mano,
# replicando cio' che fa PrefetchLoader ma senza torch.cuda.Stream/.cuda().

class ManualNormalizeLoader:
    def __init__(self, loader, device, rgb_mean, rgb_std, thermal_mean, thermal_std):
        self.loader = loader
        self.device = device
        self.rgb_mean = torch.tensor([x * 255 for x in rgb_mean]).to(device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([x * 255 for x in rgb_std]).to(device).view(1, 3, 1, 1)
        self.thermal_mean = torch.tensor([x * 255 for x in thermal_mean]).to(device).view(1, 3, 1, 1)
        self.thermal_std = torch.tensor([x * 255 for x in thermal_std]).to(device).view(1, 3, 1, 1)

    def __iter__(self):
        for thermal_input, rgb_input, target in self.loader:
            thermal_input = thermal_input.to(self.device).float().sub_(self.thermal_mean).div_(self.thermal_std)
            rgb_input = rgb_input.to(self.device).float().sub_(self.rgb_mean).div_(self.rgb_std)
            target = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                      for k, v in target.items()}
            yield thermal_input, rgb_input, target

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def build_loader(dataset, input_config, args, device):
    """DataLoader col transform numpy + DetectionFastCollate, normalizzato a mano."""
    input_size = input_config['input_size']
    img_size = input_size[-2:] if isinstance(input_size, tuple) else input_size
    dataset.transform = transforms_eval(
        img_size,
        interpolation=input_config['interpolation'],
        use_prefetcher=True,          # -> ImageToNumpy (uint8), niente CUDA qui
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'],
        thermal_std=input_config['thermal_std'])
    base = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=DetectionFastCollate(),
    )
    return ManualNormalizeLoader(
        base, device,
        rgb_mean=input_config['rgb_mean'], rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'], thermal_std=input_config['thermal_std'])


# ─────────────────────────────────────────────────────────────────────────────
# Argomenti (ricalca validate_fusion.py + opzioni di ablation)
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Modality ablation on a trained fusion model')
parser.add_argument('root', metavar='DIR', help='path to dataset root')
parser.add_argument('--branch', default='fusion', type=str,
                    help='inference branch (qui ha senso solo "fusion")')
parser.add_argument('--dataset', default='flir_aligned', type=str)
parser.add_argument('--split', default='test')
parser.add_argument('--model', '-m', metavar='MODEL', default='efficientdetv2_dt')
parser.add_argument('--num-classes', type=int, default=None)
parser.add_argument('--att_type', default='cbam', type=str, choices=['cbam', 'shuffle', 'eca'])
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--img-size', default=None, type=int)
parser.add_argument('--rgb_mean', type=float, nargs='+', default=None)
parser.add_argument('--rgb_std', type=float, nargs='+', default=None)
parser.add_argument('--thermal_mean', type=float, nargs='+', default=None)
parser.add_argument('--thermal_std', type=float, nargs='+', default=None)
parser.add_argument('--interpolation', default='bilinear', type=str)
parser.add_argument('--fill-color', default=None, type=str)
parser.add_argument('--log-freq', default=50, type=int)
parser.add_argument('--checkpoint', default='', type=str, required=True,
                    help='checkpoint del modello di fusione addestrato')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--channels', default=128, type=int)
parser.add_argument('--no-prefetcher', action='store_true', default=False)
parser.add_argument('--pin-mem', action='store_true', default=False)
parser.add_argument('--use-ema', dest='use_ema', action='store_true')
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--classwise', dest='classwise', action='store_true')
parser.add_argument('--results', default='', type=str)
parser.add_argument('--init-fusion-head-weights', type=str, default=None,
                    choices=['thermal', 'rgb', None])
parser.add_argument('--thermal-checkpoint-path', type=str, default=None)
parser.add_argument('--rgb-checkpoint-path', type=str, default=None)
parser.add_argument('--cbam-backend', type=str, default='mlp',
                    choices=["mlp", "FastKAN", "EfficientKAN", "WavKAN", "EfficientKAN_dual"])
parser.add_argument('--cbam-num-grids', type=int, default=5)
parser.add_argument('--cbam-reduction', type=int, default=16)

# ── opzioni specifiche dell'ablation ──
parser.add_argument('--ablate-mode', type=str, default='zero', choices=['zero', 'mean'],
                    help='come spegnere il ramo: "zero" (semplice, OOD) o '
                         '"mean" (media per-canale sul dataset, piu pulita)')
parser.add_argument('--conditions', type=str, nargs='+',
                    default=['full', 'no_thermal', 'no_rgb'],
                    choices=['full', 'no_thermal', 'no_rgb'],
                    help='condizioni da valutare')
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'],
                    help='device (default: auto -> cuda se disponibile, altrimenti cpu)')
parser.add_argument('--fusion-ops', type=str, nargs='+', default=None,
                    choices=['mean', 'sum', 'max', 'thermal', 'rgb'],
                    help='ANALISI B (fusione ingenua): sostituisce il CBAM appreso '
                         'con una combinazione senza parametri dei due rami. Se '
                         'impostato, valuta full(learned) + le op scelte invece '
                         'dell ablation di modalita.')
parser.add_argument('--corrupt', type=str, default=None, choices=['thermal', 'rgb'],
                    help='ESP. 2 (stress test): degrada la modalita scelta sull INPUT '
                         'e confronta full / corrupt_<mod> / no_<mod>. Se corrupt fa '
                         'piu male di no (ignorare), la fusione statica non sa '
                         'sopprimere la modalita corrotta -> fragilita.')
parser.add_argument('--corrupt-type', type=str, default='noise',
                    choices=['noise', 'dark'],
                    help='tipo di degrado: noise (gaussiano) o dark (attenuazione)')
parser.add_argument('--corrupt-strength', type=float, default=1.5,
                    help='intensita del degrado (noise: std del rumore; dark: frazione '
                         'di segnale rimossa, 0-1)')


# ─────────────────────────────────────────────────────────────────────────────
# Ablation via forward-hook sull'FPN
# ─────────────────────────────────────────────────────────────────────────────

def _make_ablation_hook(mode, means=None):
    """
    Forward-hook che sostituisce l'output dell'FPN (lista di tensori [B,C,H,W],
    uno per livello) per spegnere quella modalita' prima della fusione.
    """
    def hook(module, inputs, output):
        if mode == 'zero':
            return [torch.zeros_like(o) for o in output]
        # mode == 'mean'
        new_out = []
        for lvl, o in enumerate(output):
            m = means[lvl].to(o.device, o.dtype).view(1, -1, 1, 1)
            new_out.append(m.expand_as(o).contiguous())
        return new_out
    return hook


def _make_cbam_naive_hook(op, channels):
    """
    Forward-hook sul modulo CBAM: sostituisce l'output APPRESO (combine 2C->C +
    channel/spatial attention) con una combinazione SENZA PARAMETRI delle due
    meta' [thermal | rgb] dell'input concatenato (ciascuna C canali). Misura
    quanto aggiunge la fusione *appresa* sopra le pure feature EfficientDet.
    """
    C = channels

    def hook(module, inputs, output):
        x = inputs[0]                 # [B, 2C, H, W] = [thermal | rgb]
        tx = x[:, :C]
        vx = x[:, C:2 * C]
        if op == 'mean':
            return (tx + vx) / 2
        if op == 'sum':
            return tx + vx
        if op == 'max':
            return torch.maximum(tx, vx)
        if op == 'thermal':
            return tx
        if op == 'rgb':
            return vx
        raise ValueError(f'op di fusione ingenua sconosciuta: {op}')
    return hook


def install_hooks(bench, condition, args, thermal_means=None, rgb_means=None):
    """Installa gli hook per la condizione richiesta, ritorna la lista di handle.

    Condizioni di ablation di MODALITA' (hook sull'FPN):
        full | no_thermal | no_rgb
    Condizioni di FUSIONE INGENUA (hook sull'output dei CBAM):
        fuse_mean | fuse_sum | fuse_max | fuse_thermal | fuse_rgb
    """
    handles = []
    if condition in ('full', 'learned'):
        pass
    elif condition == 'no_thermal':
        means = thermal_means if args.ablate_mode == 'mean' else None
        handles.append(bench.model.thermal_fpn.register_forward_hook(
            _make_ablation_hook(args.ablate_mode, means)))
    elif condition == 'no_rgb':
        means = rgb_means if args.ablate_mode == 'mean' else None
        handles.append(bench.model.rgb_fpn.register_forward_hook(
            _make_ablation_hook(args.ablate_mode, means)))
    elif condition.startswith('fuse_'):
        op = condition[len('fuse_'):]
        for i in range(bench.model.config.num_levels):
            cbam = getattr(bench.model, f'fusion_{args.att_type}{i}', None)
            if cbam is not None:
                handles.append(cbam.register_forward_hook(
                    _make_cbam_naive_hook(op, args.channels)))
    elif condition.startswith('corrupt_'):
        pass   # il degrado e' applicato sull'input nel loop di evaluate_condition
    else:
        raise ValueError(f'condizione sconosciuta: {condition}')
    return handles


def _corrupt(tensor, ctype, strength):
    """Degrada un tensore di input (gia' normalizzato). 'noise' aggiunge rumore
    gaussiano (std=strength); 'dark' attenua il segnale verso lo zero/media."""
    if ctype == 'noise':
        return tensor + strength * torch.randn_like(tensor)
    elif ctype == 'dark':
        return tensor * max(0.0, 1.0 - strength)
    return tensor


def compute_fpn_means(bench, loader, amp_autocast):
    """
    Primo passaggio sul dataset: media per-canale (su batch e spazio) delle
    feature in uscita da thermal_fpn e rgb_fpn, per ogni livello FPN.
    Ritorna (thermal_means, rgb_means): dict {lvl: tensor[C]} ciascuno.
    """
    stats = {'thermal': {}, 'rgb': {}}

    def make_collect(which):
        s = stats[which]
        def hook(module, inputs, output):
            for lvl, o in enumerate(output):
                csum = o.sum(dim=(0, 2, 3)).detach().double()           # [C]
                cnt = o.shape[0] * o.shape[2] * o.shape[3]
                if lvl not in s:
                    s[lvl] = [csum, cnt]
                else:
                    s[lvl][0] += csum
                    s[lvl][1] += cnt
        return hook

    h_t = bench.model.thermal_fpn.register_forward_hook(make_collect('thermal'))
    h_r = bench.model.rgb_fpn.register_forward_hook(make_collect('rgb'))

    print('  [mean] primo passaggio per le medie per-canale dell\'FPN...')
    with torch.no_grad():
        for thermal_input, rgb_input, target in loader:
            with amp_autocast('cuda'):
                bench(thermal_input, rgb_input, img_info=target, branch='fusion')

    h_t.remove()
    h_r.remove()

    thermal_means = {lvl: (v[0] / v[1]).float() for lvl, v in stats['thermal'].items()}
    rgb_means = {lvl: (v[0] / v[1]).float() for lvl, v in stats['rgb'].items()}
    return thermal_means, rgb_means


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione di una singola condizione
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(bench, loader, dataset, args, condition,
                       amp_autocast, thermal_means=None, rgb_means=None):
    """Installa gli hook per la condizione, valuta, li rimuove, ritorna la metrica."""
    handles = install_hooks(bench, condition, args,
                            thermal_means=thermal_means, rgb_means=rgb_means)

    evaluator = create_evaluator(args.dataset + "_eval", dataset,
                                 distributed=False, pred_yxyx=False,
                                 classwise=args.classwise)
    eval_kind = type(evaluator).__name__

    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1
    output = None

    with torch.no_grad():
        for i, (thermal_input, rgb_input, target) in enumerate(loader):
            if condition == 'corrupt_rgb':
                rgb_input = _corrupt(rgb_input, args.corrupt_type, args.corrupt_strength)
            elif condition == 'corrupt_thermal':
                thermal_input = _corrupt(thermal_input, args.corrupt_type, args.corrupt_strength)
            with amp_autocast('cuda'):
                output = bench(thermal_input, rgb_input, img_info=target, branch='fusion')
            evaluator.add_predictions(output, target)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.log_freq == 0 or i == last_idx:
                print('  [{cond:>10s}] Test: [{0:>4d}/{1}]  '
                      'Time {bt.avg:.3f}s ({rate:>6.1f}/s)'.format(
                          i, len(loader), cond=condition, bt=batch_time,
                          rate=thermal_input.size(0) / batch_time.avg))

    for h in handles:
        h.remove()

    mean_ap = 0.
    ap50 = None
    if dataset.parser.has_labels:
        res_file = ''
        if args.results:
            res_file = args.results.replace('.json', f'_{condition}.json')
        mean_ap = evaluator.evaluate(output_result_file=res_file)
        # AP@0.50 ("Pascal VOC"): stats[1] dell'evaluator COCO, se disponibile
        stats = getattr(evaluator, 'stats', None)
        if stats is not None and len(stats) > 1:
            ap50 = float(stats[1])
    return float(mean_ap), ap50, eval_kind


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def autodetect_kan_config(checkpoint_path, backend, channels):
    """Per i backend KAN legge num_grids/reduction direttamente dal checkpoint,
    cosi' non serve indovinarli (un mismatch farebbe fallire silenziosamente il
    caricamento dei pesi di fusione con strict=False). Ritorna (num_grids, reduction)
    o (None, None) se non rilevabili."""
    if backend == 'mlp':
        return None, None
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    ng = red = None
    if backend in ('FastKAN',):
        gk = next((k for k in sd if k.endswith('rbf.grid')), None)
        if gk is not None:
            ng = int(sd[gk].shape[0])
        bk = next((k for k in sd if k.endswith('layers.0.base_linear.weight')), None)
        if bk is not None:
            hidden = sd[bk].shape[0]            # = (2*channels)//reduction
            if hidden > 0:
                red = (2 * channels) // hidden
    return ng, red


def assert_fusion_weights_loaded(model, checkpoint_path):
    """Backstop: verifica che i pesi 'fusion_*' del checkpoint combacino col modello.
    Con strict=False un mismatch di forma (es. num_grids errato) verrebbe scartato
    in silenzio lasciando i moduli a init random -> metriche prive di senso."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    msd = model.state_dict()
    mism = [(k, tuple(v.shape), tuple(msd[k].shape))
            for k, v in sd.items()
            if k.startswith('fusion_') and k in msd
            and tuple(msd[k].shape) != tuple(v.shape)]
    if mism:
        lines = '\n'.join(f'    {k}: ckpt={cs} vs model={ms}' for k, cs, ms in mism[:6])
        raise RuntimeError(
            f"{len(mism)} pesi di fusione NON caricati per mismatch di forma "
            f"(probabile --cbam-num-grids/--cbam-reduction errato):\n{lines}\n"
            "Sarebbero rimasti a init random -> risultati invalidi.")


def main():
    args = parser.parse_args()
    setup_default_logging()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Dataset: {args.dataset} | split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backend: {args.cbam_backend} | att_type: {args.att_type}")
    print(f"Ablate mode: {args.ablate_mode} | conditions: {args.conditions}")
    print(f"Device: {device}")

    # ── Autodetect num_grids/reduction dal checkpoint (backend KAN) ───────────
    if args.cbam_backend != 'mlp':
        ng, red = autodetect_kan_config(args.checkpoint, args.cbam_backend, args.channels)
        if ng and ng != args.cbam_num_grids:
            print(f"[auto] cbam_num_grids {args.cbam_num_grids} -> {ng} (dal checkpoint)")
            args.cbam_num_grids = ng
        if red and red != args.cbam_reduction:
            print(f"[auto] cbam_reduction {args.cbam_reduction} -> {red} (dal checkpoint)")
            args.cbam_reduction = red

    # ── Modello + checkpoint ─────────────────────────────────────────────────
    model = Att_FusionNet(args)
    load_checkpoint(model, args.checkpoint, use_ema=args.use_ema,
                    strict=False, weights_only=False)
    assert_fusion_weights_loaded(model, args.checkpoint)
    bench = DetBenchPredictImagePair(model)
    model_config = bench.config
    bench = bench.to(device)
    bench.eval()

    amp_autocast = suppress
    if args.amp and device == 'cuda':
        amp_autocast = torch.amp.autocast
        print('Using native Torch AMP.')

    # ── Dataset + loader (device-agnostico) ──────────────────────────────────
    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = build_loader(dataset, input_config, args, device)

    # ── Determina le condizioni e il tipo di analisi ─────────────────────────
    if args.corrupt:
        analysis = 'corruption'
        conditions = ['full', f'corrupt_{args.corrupt}', f'no_{args.corrupt}']
    elif args.fusion_ops:
        analysis = 'naive-fusion'
        conditions = ['full'] + [f'fuse_{op}' for op in args.fusion_ops]
    else:
        analysis = 'modality'
        conditions = list(args.conditions)

    # ── Medie FPN (solo per ablation di modalita' in modalita' "mean") ────────
    thermal_means = rgb_means = None
    need_mean = (analysis == 'modality' and args.ablate_mode == 'mean' and
                 any(c in ('no_thermal', 'no_rgb') for c in conditions))
    if need_mean:
        thermal_means, rgb_means = compute_fpn_means(bench, loader, amp_autocast)

    # ── Valutazione per condizione ───────────────────────────────────────────
    results = {}    # cond -> metrica primaria
    results_ap50 = {}   # cond -> AP@0.50 (solo CocoEvaluator)
    eval_kind = None
    for cond in conditions:
        print(f"\n=== Condizione: {cond} ===")
        m, ap50, eval_kind = evaluate_condition(
            bench, loader, dataset, args, cond, amp_autocast,
            thermal_means=thermal_means, rgb_means=rgb_means)
        results[cond] = m
        results_ap50[cond] = ap50

    # Etichetta della metrica primaria a seconda dell'evaluator usato.
    has_ap50 = any(v is not None for v in results_ap50.values())
    if has_ap50:
        metric_label = 'mAP[.5:.95]'
    elif eval_kind == 'PascalEvaluator':
        metric_label = 'mAP@0.50(VOC)'
    else:
        metric_label = f'metric({eval_kind})'

    LABELS = {
        'full': '(T+RGB, learned)', 'no_thermal': '(solo RGB)', 'no_rgb': '(solo T)',
        'fuse_mean': '(media T,RGB)', 'fuse_sum': '(somma)', 'fuse_max': '(max)',
        'fuse_thermal': '(bypass: solo T)', 'fuse_rgb': '(bypass: solo RGB)',
        'corrupt_thermal': f'(T degradato: {args.corrupt_type})',
        'corrupt_rgb': f'(RGB degradato: {args.corrupt_type})',
    }
    title = {'modality': 'ABLATION DI MODALITA',
             'naive-fusion': 'FUSIONE INGENUA (CBAM appreso sostituito, no params)',
             'corruption': f'STRESS TEST CORRUZIONE ({args.corrupt}, {args.corrupt_type})'}[analysis]

    def _dp(val, ref):
        if ref is None or val is None or ref <= 1e-9:
            return '-'
        return f'{(val - ref) / ref * 100:.1f}%'

    # ── Tabella riassuntiva ──────────────────────────────────────────────────
    base = results.get('full', None)
    base50 = results_ap50.get('full', None)
    print("\n" + "=" * 78)
    print(f" {title}  --  {args.dataset} ({args.split})")
    print(f" checkpoint: {args.checkpoint}")
    print(f" backend: {args.cbam_backend} | ablate-mode: {args.ablate_mode} "
          f"| evaluator: {eval_kind}")
    print("=" * 78)

    if has_ap50:
        print(f" {'condizione':<13}{metric_label:>12} {'d %':>8}   "
              f"{'AP@0.50':>9} {'d %':>8}")
    else:
        print(f" {'condizione':<13}{metric_label:>14} {'d %':>8}")
    print("-" * 78)
    for cond in conditions:
        if cond not in results:
            continue
        m = results[cond]
        lab = LABELS.get(cond, '')
        dpm = '-' if cond == 'full' else _dp(m, base)
        if has_ap50:
            a = results_ap50.get(cond)
            a_str = f'{a:.4f}' if a is not None else '   n/a'
            dpa = '-' if cond == 'full' else _dp(a, base50)
            print(f" {cond:<13}{m:>12.4f} {dpm:>8}   {a_str:>9} {dpa:>8}  {lab}")
        else:
            print(f" {cond:<13}{m:>14.4f} {dpm:>8}  {lab}")
    print("=" * 78)

    # ── Verdetti ──────────────────────────────────────────────────────────────
    if analysis == 'modality' and base is not None \
            and 'no_thermal' in results and 'no_rgb' in results:
        drop_t = base - results['no_thermal']
        drop_r = base - results['no_rgb']
        print(f" Calo {metric_label} togliendo THERMAL/Gated: {drop_t:.4f}   "
              f"togliendo RGB: {drop_r:.4f}")
        if max(drop_t, drop_r) > 1e-6:
            who = 'THERMAL/Gated' if drop_t > drop_r else 'RGB'
            print(f" -> La rete dipende di piu' da: {who}")
        print("=" * 78)
    elif analysis == 'naive-fusion' and base is not None:
        best = max((c for c in conditions if c != 'full'),
                   key=lambda c: results.get(c, -1), default=None)
        if best is not None:
            gap = base - results[best]
            print(f" CBAM appreso (full) = {base:.4f} {metric_label}")
            print(f" Migliore fusione ingenua = {best} -> {results[best]:.4f} "
                  f"({_dp(results[best], base)} vs learned)")
            print(f" Valore aggiunto della fusione APPRESA: {gap:+.4f} {metric_label}")
        print("=" * 78)
    elif analysis == 'corruption' and base is not None:
        c_corr = results.get(f'corrupt_{args.corrupt}')
        c_no = results.get(f'no_{args.corrupt}')
        if c_corr is not None and c_no is not None:
            print(f" full={base:.4f}  corrupt_{args.corrupt}={c_corr:.4f}  "
                  f"no_{args.corrupt}={c_no:.4f}")
            if c_corr < c_no - 1e-6:
                print(f" -> {args.corrupt} CORROTTO fa PIU male che IGNORARLO "
                      f"(corrupt {c_corr:.4f} < no {c_no:.4f}): la fusione statica NON "
                      "sa sopprimere la modalita degradata -> FRAGILITA (motiva l'adattivo).")
            else:
                print(f" -> la rete tollera il degrado di {args.corrupt} "
                      f"(corrupt >= no): poco sensibile a questa modalita qui.")
        print("=" * 78)


if __name__ == '__main__':
    main()
