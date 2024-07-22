from nilearn.image import resample_to_img, new_img_like, index_img
from nilearn import image, masking, plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import itertools
from pathlib import Path
import pandas as pd


class DataLoader:
    def __init__(self,source_dir):
        subfolders = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]
        self.filtered_subfolders = [code for code in subfolders if code.startswith('sub-')]

    def get_subject_codes(self):
        return self.filtered_subfolders

    def load_fmri(self, subject_code):
        fmri_file_path = f'/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep/{subject_code}/ses-01/func/{subject_code}_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        fmri_img = image.load_img(fmri_file_path)
        return fmri_img

    def apply_gray_matter_mask(self, fmri_img):
        # 이진 회색질 마스크를 fMRI 데이터에 적용
        # Load the gray matter mask and convert to binary
        gray_matter_mask_path = "/home/changbae/fmri_project/MDMR/template/tpl-MNI152NLin2009cAsym_space-MNI_res-01_class-GM_probtissue.nii"
        gray_matter_mask = image.load_img(gray_matter_mask_path)
        resampled_gray_matter_mask = resample_to_img(gray_matter_mask, self.load_fmri(self.filtered_subfolders[0]), interpolation='nearest')
        gray_matter_mask_data = resampled_gray_matter_mask.get_fdata()
        binary_gray_matter_mask_data = (gray_matter_mask_data > 0.5).astype(np.int32)
        binary_gray_matter_mask = new_img_like(resampled_gray_matter_mask, binary_gray_matter_mask_data)

        # Flatten the mask and get the 3D coordinates of the mask
        coords = np.column_stack(np.where(binary_gray_matter_mask_data > 0))

        masked_data = masking.apply_mask(fmri_img, binary_gray_matter_mask)

        self.binary_gray_matter_mask = binary_gray_matter_mask
        self.gray_matter_coords = coords
        return masked_data
    def apply_common_nonzero_mask(self, masked_data):
        common_nonzero_mask = np.load("/home/changbae/fmri_project/MDMR/template/common_mask_as_np.npy")
        nonzero_masked_data = masked_data[:, common_nonzero_mask]
        self.nonzero_coords = self.gray_matter_coords[common_nonzero_mask]

        return nonzero_masked_data
    def get_nonzero_coords(self):
        return self.nonzero_coords
class ImagePlotter:
    def __init__(self):
        pass
    def plot_original_fmri(self, fmri_data):
        fmri_img_3d = index_img(fmri_data, 0)
        # 첫 번째 시간 포인트에서 원래 fMRI 데이터 플롯
        plotting.plot_epi(fmri_img_3d, title="Original fMRI Image (0th time point)", display_mode='ortho', cut_coords=(0, 0, 0), draw_cross=True, annotate=False)
    def plot_masked_fmri(self, masked_data, binary_gray_matter_mask):
        masked_img_4d = masking.unmask(masked_data, binary_gray_matter_mask)

        # 첫 번째 시간 포인트에서 masked_data를 3D 이미지로 변환
        masked_img_3d = index_img(masked_img_4d, 0)
        print(f"Masked fMRI shape: {masked_img_3d.shape}")

        # 첫 번째 시간 포인트에서 masked_data 플롯
        plt.figure(figsize=(12, 6))
        plotting.plot_epi(masked_img_3d, title="Masked fMRI Data (0th time point)", display_mode='ortho', cut_coords=(0, 0, 0), draw_cross=True, annotate=False)
        plt.show()
    def plot_correlation(self, fmri, correlation, nonzero_coords):
        shape = fmri.shape[:3]
        correlation_3d = np.zeros(shape)
        for i, coord in enumerate(nonzero_coords):
            correlation_3d[tuple(coord)] = correlation[i]

        # 결과 시각화
        new_img = nib.Nifti1Image(correlation_3d, affine=fmri.affine)
        plotting.plot_stat_map(new_img, title="Voxel 0 Temporal Correlation", display_mode='ortho')
        plt.show()

class VoxelLoader:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        subfolders = [item.name for item in self.source_dir.iterdir() if item.is_dir()]
        self.filtered_subfolders = [code for code in subfolders if code.startswith('sub-')]
        
    def get_subject_combination(self):
        self.subject_combinations = list(itertools.combinations(self.filtered_subfolders, 2))
        return self.subject_combinations
    
    def get_voxel_count(self):
        subject_path = self.source_dir / self.filtered_subfolders[0]
        if subject_path.exists() and subject_path.is_dir():
            voxel_files = [file for file in subject_path.iterdir() if file.is_file()]
            return len(voxel_files)
        else:
            return -1
    
    def load_voxel(self, subject_code, voxel_index):
        voxel_path = self.source_dir / subject_code / f"voxel_{voxel_index}.txt"
        voxel_data = np.loadtxt(voxel_path)
        voxel_data = np.delete(voxel_data, voxel_index)
        return voxel_data