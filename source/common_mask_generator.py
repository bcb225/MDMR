from package.DataLoader import DataLoader
from nilearn.image import resample_to_img, new_img_like
from nilearn import image
import numpy as np
import os

source_dir = "/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep"
gray_matter_mask_path = "/home/changbae/fmri_project/MDMR/template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii"
output_file_path = "/home/changbae/fmri_project/MDMR/template/common_mask_as_np.npy"  # 저장할 파일 경로

data_loader = DataLoader()
subject_codes = data_loader.get_subject_codes(source_dir)
filtered_subject_codes = [code for code in subject_codes if code.startswith('sub-')]

# Load the gray matter mask and convert to binary
gray_matter_mask = image.load_img(gray_matter_mask_path)
resampled_gray_matter_mask = resample_to_img(gray_matter_mask, data_loader.load_fmri(filtered_subject_codes[0]), interpolation='nearest')
gray_matter_mask_data = resampled_gray_matter_mask.get_fdata()
binary_gray_matter_mask_data = (gray_matter_mask_data > 0.5).astype(np.int32)
binary_gray_matter_mask = new_img_like(resampled_gray_matter_mask, binary_gray_matter_mask_data)

all_masked_data = []
common_nonzero_mask = None

for subject_code in filtered_subject_codes:
    fmri_img = data_loader.load_fmri(subject_code)
    masked_data = data_loader.apply_gray_matter_mask(fmri_img, binary_gray_matter_mask)
    
    # 모든 타임 포인트에서 0이 아닌 복셀만을 포함하는 마스크 생성
    non_zero_mask = np.all(masked_data != 0, axis=0)
    
    # Combine all masked data
    all_masked_data.append(masked_data)
    
    # Identify common non-zero voxels across all subjects
    if common_nonzero_mask is None:
        common_nonzero_mask = non_zero_mask
    else:
        common_nonzero_mask &= non_zero_mask

print(f"Count of common non-zero voxels: {np.sum(common_nonzero_mask)}")
# common_nonzero_mask를 numpy 파일로 저장
np.save(output_file_path, common_nonzero_mask)
print(f"common_nonzero_mask saved to {output_file_path}")