import subprocess

backend = "FastKAN"
# DATASETS = ["flir_aligned_full", "flir_aligned_day", "flir_aligned_night"]
REDUCTIONS = [16, 32, 64]
NUM_GRIDS = [2, 3, 4]
# REDUCTIONS = [64]
# NUM_GRIDS = [4]

# for dataset in DATASETS:
for reduction in REDUCTIONS:
    for num_grids in NUM_GRIDS:
        output_folder_suffix = '_' + backend + '_' + str(reduction) + '_' + str(num_grids)

        # command = f"python validate_fusion_adaptive.py Datasets/FLIR_Aligned --dataset flir_aligned_full --num-scenes 3 --checkpoint output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar --checkpoint-cls Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --checkpoint-scenes output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_DAY_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_NIGHT_CBAM/model_best.pth.tar --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"
        command = f"python validate_fusion.py Datasets/FLIR_Aligned --dataset flir_aligned_full --checkpoint output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"
        subprocess.run(command, shell=True)
