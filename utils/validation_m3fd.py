import subprocess
import re
import csv


csvfile = open('m3fd_results.csv', 'w', newline='')
classes = ['People', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Lamp']
fieldnames = ['model', 'parameters', 'mAP@0.5IOU'] + [f'AP@0.5IOU/{c}' for c in classes]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

to_validate = [
    ('agnostic', 'm3fd_full', 'FastKAN', '64', '2'),
    ('agnostic', 'm3fd_full', 'FastKAN', '64', '3'),
    ('agnostic', 'm3fd_full', 'FastKAN', '32', '2'),
    ('agnostic', 'm3fd_full', 'FastKAN', '32', '3'),
    ('agnostic', 'm3fd_full', 'EfficientKAN_dual', '64', '2'),
    ('agnostic', 'm3fd_full', 'EfficientKAN_dual', '64', '3'),
    ('agnostic', 'm3fd_full', 'EfficientKAN_dual', '32', '2'),
    ('agnostic', 'm3fd_full', 'EfficientKAN_dual', '32', '3'),
    ('agnostic', 'm3fd_full', 'EfficientKAN', '64', '2'),
    ('agnostic', 'm3fd_full', 'EfficientKAN', '64', '3'),
    ('agnostic', 'm3fd_full', 'EfficientKAN', '32', '2'),
    ('agnostic', 'm3fd_full', 'EfficientKAN', '32', '3'),
    ('adaptive', 'm3fd_full', 'FastKAN', '64', '2'),
    ('adaptive', 'm3fd_full', 'FastKAN', '64', '3'),
    ('adaptive', 'm3fd_full', 'FastKAN', '32', '2'),
    ('adaptive', 'm3fd_full', 'FastKAN', '32', '3'),
    ('adaptive', 'm3fd_full', 'EfficientKAN_dual', '64', '2'),
    ('adaptive', 'm3fd_full', 'EfficientKAN_dual', '64', '3'),
    ('adaptive', 'm3fd_full', 'EfficientKAN_dual', '32', '2'),
    ('adaptive', 'm3fd_full', 'EfficientKAN_dual', '32', '3'),
    ('adaptive', 'm3fd_full', 'EfficientKAN', '64', '2'),
    ('adaptive', 'm3fd_full', 'EfficientKAN', '64', '3'),
    ('adaptive', 'm3fd_full', 'EfficientKAN', '32', '2'),
    ('adaptive', 'm3fd_full', 'EfficientKAN', '32', '3')
]

for fusion_type, dataset, backend, reduction, num_grids in to_validate:
    print(fusion_type, dataset, backend, reduction, num_grids)
    output_folder_suffix = '_' + backend + '_' + str(reduction) + '_' + str(num_grids)

    if fusion_type == "adaptive":
        command = f"python validate_fusion_adaptive.py Datasets/M3FD --dataset {dataset} --num-scenes 5 --checkpoint output{output_folder_suffix}_M3FD/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --checkpoint-cls Checkpoints/M3FD/Classifier/m3fd_classifier.pth.tar --checkpoint-scenes output{output_folder_suffix}_M3FD/train_flir/EXP_M3FD_FULL_CBAM/model_best.pth.tar output{output_folder_suffix}_M3FD/train_flir/EXP_M3FD_DAY_CBAM/model_best.pth.tar output{output_folder_suffix}_M3FD/train_flir/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar output{output_folder_suffix}_M3FD/train_flir/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar output{output_folder_suffix}_M3FD/train_flir/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --classwise --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"
    elif fusion_type == "agnostic":
        command = f"python validate_fusion.py Datasets/M3FD --dataset {dataset} --checkpoint output{output_folder_suffix}_M3FD/train_flir/EXP_{dataset.upper()}_CBAM/model_best.pth.tar --classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --cbam-backend {backend} --cbam-reduction {reduction} --cbam-num-grids {num_grids}"

    output = subprocess.run(command, shell=True, capture_output=True)
    output = output.stdout.decode() + output.stderr.decode()

    model = output_folder_suffix
    parameters = re.search(r'^Parameters:\s*(\d+)', output, re.MULTILINE).group(1)
    mAP50 = re.search(r'^PascalBoxes_Precision/mAP@0\.5IOU:\s*([\d.]+)', output, re.MULTILINE).group(1)

    row = {'model': model, 'parameters': parameters, 'mAP@0.5IOU': mAP50}

    for c in classes:
        mAP50_c = re.search(r'^PascalBoxes_PerformanceByCategory/AP@0\.5IOU/' + c + r':\s*([\d.]+)', output, re.MULTILINE).group(1)
        row['AP@0.5IOU/'+c] = mAP50_c

    writer.writerow(row)

csvfile.close()
