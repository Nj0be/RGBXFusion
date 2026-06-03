import subprocess

# backend = "FastKAN"
# BACKENDS = ["FastKAN", "EfficientKAN_dual"]
BACKENDS = ["EfficientKAN", "WavKAN"]
# DATASETS = ["flir_aligned_full", "flir_aligned_day", "flir_aligned_night"]
DATASETS = ["m3fd_day", "m3fd_night", "m3fd_overcast", "m3fd_challenge", "m3fd_full"]
# REDUCTIONS = [16, 32, 64, 128]
REDUCTIONS = [32, 64]
# NUM_GRIDS = [2, 3, 4]
NUM_GRIDS = [2, 3]

for dataset in DATASETS:
    for backend in BACKENDS:
        for reduction in REDUCTIONS:
            for num_grids in NUM_GRIDS:

                output_folder_suffix = '_' + backend + '_' + str(reduction) + '_' + str(num_grids)

                # command = f"python train_fusion.py Datasets/FLIR_Aligned --dataset {dataset} --thermal-checkpoint-path Checkpoints/FLIR_Aligned/Single_Modality_Models/flir_thermal_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids} --output output{output_folder_suffix}"
                command = f"python train_fusion.py Datasets/M3FD --dataset {dataset} --rgb-checkpoint-path Checkpoints/M3FD/Single_Modality_Models/m3fd_rgb_backbone.pth.tar --thermal-checkpoint-path Checkpoints/M3FD/Single_Modality_Models/m3fd_thermal_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids} --output output{output_folder_suffix}_M3FD"
                subprocess.run(command, shell=True)
