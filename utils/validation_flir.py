import subprocess
import re
import csv


csvfile = open('flir_results.csv', 'w', newline='')
classes = ['person', 'bike', 'car']
fieldnames = ['model', 'parameters', 'mAP@0.5IOU'] + [f'AP@0.5IOU/{c}' for c in classes]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

'''
'''
# ('agnostic', 'flir_aligned_full', 'EfficientKAN_dual', '64', '3'),
# ('agnostic', 'flir_aligned_full', 'EfficientKAN_dual', '32', '3'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN_dual', '64', '3'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN_dual', '32', '3'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN', '64', '2'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN', '64', '3'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN', '32', '2'),
# ('adaptive', 'flir_aligned_full', 'EfficientKAN', '32', '3')
to_validate = [
    ('agnostic', 'flir_aligned_full', 'FastKAN', '64', '2'),
    ('agnostic', 'flir_aligned_full', 'FastKAN', '64', '3'),
    ('agnostic', 'flir_aligned_full', 'FastKAN', '32', '2'),
    ('agnostic', 'flir_aligned_full', 'FastKAN', '32', '3'),
    ('agnostic', 'flir_aligned_full', 'EfficientKAN_dual', '64', '2'),
    ('agnostic', 'flir_aligned_full', 'EfficientKAN_dual', '32', '2'),
    ('adaptive', 'flir_aligned_full', 'FastKAN', '64', '2'),
    ('adaptive', 'flir_aligned_full', 'FastKAN', '64', '3'),
    ('adaptive', 'flir_aligned_full', 'FastKAN', '32', '2'),
    ('adaptive', 'flir_aligned_full', 'FastKAN', '32', '3'),
    ('adaptive', 'flir_aligned_full', 'EfficientKAN_dual', '64', '2'),
    ('adaptive', 'flir_aligned_full', 'EfficientKAN_dual', '32', '2')
]

for fusion_type, dataset, backend, reduction, num_grids in to_validate:
    print(fusion_type, dataset, backend, reduction, num_grids)
    output_folder_suffix = '_' + backend + '_' + str(reduction) + '_' + str(num_grids)

    if fusion_type == "adaptive":
        command = f"python validate_fusion_adaptive.py Datasets/FLIR_Aligned --dataset {dataset} --num-scenes 3 --checkpoint output{output_folder_suffix}/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --checkpoint-cls Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --checkpoint-scenes output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_DAY_CBAM/model_best.pth.tar output{output_folder_suffix}/train_flir/EXP_FLIR_ALIGNED_NIGHT_CBAM/model_best.pth.tar --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"
    elif fusion_type == "agnostic":
        command = f"python validate_fusion.py Datasets/FLIR_Aligned --dataset {dataset} --checkpoint output{output_folder_suffix}/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"

    output = subprocess.run(command, shell=True, capture_output=True)
    output = output.stdout.decode() + output.stderr.decode()

    print(output)

    model = output_folder_suffix
    parameters = re.search(r'^Parameters:\s*(\d+)', output, re.MULTILINE).group(1)
    mAP50 = re.search(r'^PascalBoxes_Precision/mAP@0\.5IOU:\s*([\d.]+)', output, re.MULTILINE).group(1)

    row = {'model': model, 'parameters': parameters, 'mAP@0.5IOU': mAP50}

    for c in classes:
        mAP50_c = re.search(r'^PascalBoxes_PerformanceByCategory/AP@0\.5IOU/' + c + r':\s*([\d.]+)', output, re.MULTILINE).group(1)
        row['AP@0.5IOU/'+c] = mAP50_c

    writer.writerow(row)

csvfile.close()
