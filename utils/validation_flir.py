import subprocess
import re
import csv

DATASETS = ["flir_aligned_full", "flir_aligned_day", "flir_aligned_night"]

csvfile = open(f'results_flir.csv', 'w', newline='')
classes = ['person', 'bike', 'car']
fieldnames = ['dataset', 'model', 'parameters', 'Pascal mAP@0.5IOU'] + [f'Pascal AP@0.5IOU/{c}' for c in classes] + [
    'COCO AP@[0.50:0.95]',
    'COCO AP@0.50',
    'COCO AP@0.75',
    'COCO AP_small',
    'COCO AP_medium',
    'COCO AP_large',
    'COCO AR@1',
    'COCO AR@10',
    'COCO AR@100',
    'COCO AR_small',
    'COCO AR_medium',
    'COCO AR_large',
    'COCO mAP'
]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

for dataset in DATASETS:
    to_validate = [
        ('agnostic', 'mlp', '', ''),
        ('agnostic', 'FastKAN', '32', '3'),
        ('agnostic', 'FastKAN', '32', '2'),
        ('agnostic', 'FastKAN', '64', '3'),
        ('agnostic', 'FastKAN', '64', '2'),
        ('agnostic', 'EfficientKAN', '32', '3'),
        ('agnostic', 'EfficientKAN', '32', '2'),
        ('agnostic', 'EfficientKAN', '64', '3'),
        ('agnostic', 'EfficientKAN', '64', '2'),
        ('agnostic', 'EfficientKAN_dual', '32', '3'),
        ('agnostic', 'EfficientKAN_dual', '32', '2'),
        ('agnostic', 'EfficientKAN_dual', '64', '3'),
        ('agnostic', 'EfficientKAN_dual', '64', '2'),
        ('adaptive', 'mlp', '', ''),
        ('adaptive', 'FastKAN', '32', '3'),
        ('adaptive', 'FastKAN', '32', '2'),
        ('adaptive', 'FastKAN', '64', '3'),
        ('adaptive', 'FastKAN', '64', '2'),
        ('adaptive', 'EfficientKAN', '32', '3'),
        ('adaptive', 'EfficientKAN', '32', '2'),
        ('adaptive', 'EfficientKAN', '64', '3'),
        ('adaptive', 'EfficientKAN', '64', '2'),
        ('adaptive', 'EfficientKAN_dual', '32', '3'),
        ('adaptive', 'EfficientKAN_dual', '32', '2'),
        ('adaptive', 'EfficientKAN_dual', '64', '3'),
        ('adaptive', 'EfficientKAN_dual', '64', '2'),
    ]

    for fusion_type, backend, reduction, num_grids in to_validate:
        print(fusion_type, dataset, backend, reduction, num_grids)
        output_folder_suffix = '_' + backend + '_' + str(reduction) + '_' + str(num_grids)
        model = output_folder_suffix
        if backend == "mlp":
            model = "mlp"

        if backend == "mlp":
            if fusion_type == "adaptive":
                command = "python validate_fusion_adaptive.py Datasets/FLIR_Aligned --dataset flir_aligned_full --num-scenes 3 --checkpoint Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar --checkpoint-cls Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --checkpoint-scenes Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar Checkpoints/FLIR_Aligned/Fusion_Models/Day/model_best.pth.tar Checkpoints/FLIR_Aligned/Fusion_Models/Night/model_best.pth.tar --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam"
            elif fusion_type == "agnostic":
                command = "python validate_fusion.py Datasets/FLIR_Aligned --dataset flir_aligned_full --checkpoint Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam"
        else:
            if fusion_type == "adaptive":
                command = f"python validate_fusion_adaptive.py Datasets/FLIR_Aligned --dataset {dataset} --num-scenes 3 --checkpoint output{output_folder_suffix}/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --checkpoint-cls Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --checkpoint-scenes output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_DAY_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_NIGHT_CBAM/model_best.pth.tar --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"
            elif fusion_type == "agnostic":
                command = f"python validate_fusion.py Datasets/FLIR_Aligned --dataset {dataset} --checkpoint output{output_folder_suffix}/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"

        output = subprocess.run(command+" --classwise", shell=True, capture_output=True)
        output = output.stdout.decode() + output.stderr.decode()

        parameters = re.search(r'^Parameters:\s*(\d+)', output, re.MULTILINE).group(1) 
        pascal_mAP50 = re.search(r'^PascalBoxes_Precision/mAP@0\.5IOU:\s*([\d.]+)', output, re.MULTILINE).group(1)

        row = {'dataset': dataset, 'model': model, 'parameters': parameters, 'Pascal mAP@0.5IOU': pascal_mAP50}

        for c in classes:
            mAP50_c = re.search(r'^PascalBoxes_PerformanceByCategory/AP@0\.5IOU/' + c + r':\s*([\d.]+)', output, re.MULTILINE).group(1)
            row['Pascal AP@0.5IOU/'+c] = mAP50_c


        output = subprocess.run(command, shell=True, capture_output=True)
        output = output.stdout.decode() + output.stderr.decode()

        row['COCO AP@[0.50:0.95]'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AP@0.50'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.50\s+\| area=\s*all \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AP@0.75'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.75\s+\| area=\s*all \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AP_small'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*small \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AP_medium'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*medium \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AP_large'] = re.search(
            r'Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*large \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR@1'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=\s*1 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR@10'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=\s*10 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR@100'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR_small'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*small \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR_medium'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*medium \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO AR_large'] = re.search(
            r'Average Recall\s+\(AR\)\s+@\[ IoU=0\.50:0\.95 \| area=\s*large \| maxDets=100 \]\s*=\s*([\d.]+)',
            output
        ).group(1)

        row['COCO mAP'] = re.search(
            r'Mean Average Precision Obtained is\s*:\s*([\d.]+)',
            output
        ).group(1)

        writer.writerow(row)

csvfile.close()
