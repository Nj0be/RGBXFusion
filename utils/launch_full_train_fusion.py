import subprocess

BACKENDS = ["mlp", "FastKAN", "EfficientKAN", "WavKAN"]
DATASETS = ["flir_aligned_full", "flir_aligned_day", "flir_aligned_night"]

for backend in BACKENDS:
    for dataset in DATASETS:
        output_folder_suffix = '' if backend == 'mlp' else '_'+backend

        command = f"python train_fusion.py Datasets/FLIR_Aligned --dataset {dataset} --thermal-checkpoint-path Checkpoints/FLIR_Aligned/Single_Modality_Models/flir_thermal_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam --cbam-backend {backend} --output output{output_folder_suffix}"
        subprocess.run(command, shell=True)
