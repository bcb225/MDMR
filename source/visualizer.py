import pandas as pd
import numpy as np
import argparse
import numpy as np
import pandas as pd
from nilearn import masking, plotting, image
from nilearn.image import resample_to_img, new_img_like
import os
from package.DataLoader import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run permutation test on fMRI data")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor")
    parser.add_argument("--type", type=str, required=True, help="Type")
    args = parser.parse_args()

    predictor = pd.read_csv(f"../result/fstat/{args.predictor}.csv")
    voxel_3d_index = pd.read_csv("../template/common_mask_as_np_coords.csv")
    
    # Set the starting voxel, only for temporary (before full calculation of distance matrix)
    last_line = predictor.tail(1)
    start_voxel = int(last_line['voxel'].values[0]) + 1
    end_voxel = 13718

    # Generate random data
    random_data = {
        'voxel': np.arange(start_voxel, end_voxel + 1),
        'f_stat': np.random.rand(end_voxel - start_voxel + 1),
        'p_val': np.random.rand(end_voxel - start_voxel + 1)
    }
    # Create a DataFrame for the random data
    random_df = pd.DataFrame(random_data)

    # Append the random data to the original DataFrame
    predictor = pd.concat([predictor, random_df], ignore_index=True)
    
    merged_df = pd.merge(predictor, voxel_3d_index, left_on='voxel', right_index=True, how='left')

    # 회색질 마스크 로드 및 이진화
    gray_matter_mask_path = "/home/changbae/fmri_project/MDMR/template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii"
    gray_matter_mask = image.load_img(gray_matter_mask_path)

    # 예제 fMRI 이미지를 로드하여 동일한 해상도로 재샘플링
    data_loader = DataLoader(source_dir="/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep")
    subject_codes = data_loader.get_subject_codes()
    fmri = data_loader.load_fmri(subject_codes[0])
    subject_fmri_img = image.load_img(fmri)

    resampled_gray_matter_mask = resample_to_img(gray_matter_mask, subject_fmri_img, interpolation='nearest')
    gray_matter_mask_data = resampled_gray_matter_mask.get_fdata()
    binary_gray_matter_mask_data = (gray_matter_mask_data > 0.5).astype(np.int32)
    binary_gray_matter_mask = new_img_like(resampled_gray_matter_mask, binary_gray_matter_mask_data)

    if args.type == "f":
        # f-statistics를 위한 빈 3D 배열 초기화
        f_stat_map_data = np.zeros(binary_gray_matter_mask.shape)

        # 대응하는 값들로 3D 배열 채우기
        for idx, row in predictor.iterrows():
            x, y, z = voxel_3d_index.loc[idx, ['x', 'y', 'z']]
            f_stat_map_data[int(x), int(y), int(z)] = row['f_stat']

        
        # 3D 배열에서 새로운 이미지 객체 생성
        f_stat_map_img = new_img_like(binary_gray_matter_mask, f_stat_map_data)

        # f-statistics 맵을 인터랙티브하게 시각화
        html_view_f_stat = plotting.view_img(f_stat_map_img, threshold=0, vmax=1.8, 
                                            bg_img=gray_matter_mask, cut_coords=[0, 0, 0], 
                                            title="F-statistics Map", colorbar=True)
        html_view_f_stat.open_in_browser()
    elif args.type == "p":
        p_val_map_data = np.zeros(binary_gray_matter_mask.shape)

        # 대응하는 값들로 3D 배열 채우기
        for idx, row in predictor.iterrows():
            x, y, z = voxel_3d_index.loc[idx, ['x', 'y', 'z']]
            p_val_map_data[int(x), int(y), int(z)] = row['p_val']

        # 3D 배열에서 새로운 이미지 객체 생성
        p_val_map_img = new_img_like(binary_gray_matter_mask, p_val_map_data)

        # p-value < 0.05인 영역만 시각화하기 위해 임계값 적용
        thresholded_p_val_map_data = np.zeros_like(p_val_map_data)
        thresholded_p_val_map_data[p_val_map_data < 0.05] = p_val_map_data[p_val_map_data < 0.05]

        # 3D 배열에서 새로운 임계값이 적용된 이미지 객체 생성
        thresholded_p_val_map_img = new_img_like(p_val_map_img, thresholded_p_val_map_data)

        # 임계값이 적용된 p-value 맵을 인터랙티브하게 시각화
        html_view_p_val = plotting.view_img(thresholded_p_val_map_img, threshold=0, vmax=0.05, 
                                            bg_img=gray_matter_mask, cut_coords=[0, 0, 0], 
                                            title="P-value Map (Thresholded at 0.05)", colorbar=True)
        html_view_p_val.open_in_browser()