from package.DataLoader import DataLoader
from nilearn.image import resample_to_img, new_img_like
from nilearn import image
import numpy as np
import pandas as pd

source_dir = "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep"
gray_matter_mask_path = "/home/changbae/fmri_project/MDMR/template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii"
output_file_path = "/home/changbae/fmri_project/MDMR/template/common_mask_as_np.npy"  # 저장할 파일 경로
coords_output_csv_path = output_file_path.replace(".npy", "_coords.csv")

data_loader = DataLoader(source_dir)
subject_codes = data_loader.get_subject_codes()
filtered_subject_codes = [code for code in subject_codes if code.startswith('sub-')]

# Load the gray matter mask and convert to binary
gray_matter_mask = image.load_img(gray_matter_mask_path)

# Resample gray matter mask to 1st participant to get same resolution.
resampled_gray_matter_mask = resample_to_img(gray_matter_mask, data_loader.load_fmri(filtered_subject_codes[0]), interpolation='nearest')

# get numpy array from gray matter mask.
gray_matter_mask_data = resampled_gray_matter_mask.get_fdata()

# gray matter mask threshold, prev work done with 0.25. should check afterward
binary_gray_matter_mask_data = (gray_matter_mask_data > 0.5).astype(np.int32)

# create new image object from binary mask.
binary_gray_matter_mask = new_img_like(resampled_gray_matter_mask, binary_gray_matter_mask_data)

all_masked_data = []
common_nonzero_mask = None

for subject_code in filtered_subject_codes:
    fmri_img = data_loader.load_fmri(subject_code)
    masked_data = data_loader.apply_gray_matter_mask(fmri_img)
    
    # 모든 타임 포인트에서 0이 아닌 복셀만을 포함하는 마스크 생성
    non_zero_mask = np.all(masked_data != 0, axis=0)
    
    # Combine all masked data
    all_masked_data.append(masked_data)
    
    # Identify common non-zero voxels across all subjects
    if common_nonzero_mask is None:
        common_nonzero_mask = non_zero_mask
    else:
        common_nonzero_mask &= non_zero_mask
# common_nonzero_mask의 non-zero 좌표 추출
fmri = data_loader.load_fmri(subject_code=filtered_subject_codes[0])
masked_data = data_loader.apply_gray_matter_mask(fmri_img=fmri)
nonzero_masked_data = data_loader.apply_common_nonzero_mask(masked_data=masked_data)
nonzero_coords = data_loader.get_nonzero_coords()
print(nonzero_coords)

print(f"Count of common non-zero voxels: {np.sum(common_nonzero_mask)}")

df_nonzero_coords = pd.DataFrame(nonzero_coords, columns=['x', 'y', 'z'])
df_nonzero_coords.to_csv(coords_output_csv_path, index=False)
print(f"Non-zero voxel coordinates saved to {coords_output_csv_path}")

# common_nonzero_mask를 numpy 파일로 저장
np.save(output_file_path, common_nonzero_mask)
print(f"common_nonzero_mask saved to {output_file_path}")