from tqdm import tqdm
from package.DataLoader import DataLoader
from package.StatHandler import StatHandler
import concurrent.futures
from pathlib import Path
import numpy as np

def process_subject(subject_code):
    data_loader = DataLoader(source_dir="/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep")
    stat_handler = StatHandler()
    
    fmri = data_loader.load_fmri(subject_code=subject_code)
    masked_data = data_loader.apply_gray_matter_mask(fmri_img=fmri)
    nonzero_masked_data = data_loader.apply_common_nonzero_mask(masked_data=masked_data)

    subject_dir = Path("/home/changbae/fmri_project/MDMR/result/correlation_matrix/") / subject_code
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    for vox in range(nonzero_masked_data.shape[1]):
        corr = stat_handler.temporal_correlation_by_voxel(masked_data=nonzero_masked_data, index=vox)
        voxel_file_path = subject_dir / f"voxel_{vox}.txt"
        np.savetxt(voxel_file_path, corr)
    
    return subject_code

# Initialize data loader and get subject codes
data_loader = DataLoader(source_dir="/mnt/NAS2/data/SAD_gangnam_resting/fMRIPrep")
subject_codes = data_loader.get_subject_codes()

# Use ProcessPoolExecutor to process 20 subjects concurrently
with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
    futures = {executor.submit(process_subject, subject_code): subject_code for subject_code in subject_codes}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(subject_codes), desc="Overall Progress"):
        subject_code = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(f'Subject {subject_code} generated an exception: {exc}')
