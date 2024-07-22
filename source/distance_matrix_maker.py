from tqdm import tqdm
from package.DataLoader import DataLoader
from package.DataLoader import VoxelLoader
from package.StatHandler import StatHandler
import concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import time

voxel_loader = VoxelLoader(source_dir="/home/changbae/fmri_project/MDMR/result/correlation_matrix")
subject_combination_list = voxel_loader.get_subject_combination()
voxel_count = voxel_loader.get_voxel_count()
stat_handler = StatHandler()
output_dir = Path("/home/changbae/fmri_project/MDMR/result/distance_matrix")
output_dir.mkdir(parents=True, exist_ok=True) 

for voxel_index in tqdm(range(voxel_count), desc="Processing voxels"):
    distances = []
    for subject_tuple in subject_combination_list:
        source_dir = Path("/home/changbae/fmri_project/MDMR/result/correlation_matrix")
        subject_1 = subject_tuple[0]
        subject_2 = subject_tuple[1]

        subject_1_voxel = voxel_loader.load_voxel(
            subject_code = subject_1,
            voxel_index = voxel_index
            )
        subject_2_voxel = voxel_loader.load_voxel(
            subject_code = subject_2,
            voxel_index = voxel_index
        )
        """distance = stat_handler.calculate_distance_custom(
            corr1 = subject_1_voxel,
            corr2 = subject_2_voxel
        )"""
        distance = stat_handler.calculate_distance(
            corr1 = subject_1_voxel,
            corr2 = subject_2_voxel
        )
        distances.append([subject_1, subject_2, distance])
    df = pd.DataFrame(distances, columns=["subject_code_1", "subject_code_2", "distance"])
    output_file = output_dir / f"voxel_{voxel_index}.csv"
    df.to_csv(output_file, index=False)