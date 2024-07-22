import os
import random
import pandas as pd
from package.DataLoader import DataLoader

def generate_predictor_file(source_dir, output_file):
    data_loader = DataLoader(source_dir)
    subject_codes = data_loader.get_subject_codes()
    
    # Create a list to store the predictor data
    predictor_data = []
    
    for subject in subject_codes:
        if subject.startswith('sub-s'):
            fear_value = random.uniform(0, 1) * 2  # 환자 그룹은 0-2 사이의 랜덤 값
        else:
            fear_value = random.uniform(0, 1)  # 컨트롤 그룹은 0-1 사이의 랜덤 값
        
        predictor_data.append([subject, fear_value])
    
    # Create a DataFrame and save it to a CSV file
    predictor_df = pd.DataFrame(predictor_data, columns=['subject', 'fear'])
    predictor_df.to_csv(output_file, index=False)
    print(f"Predictor file saved to {output_file}")

# Example usage
source_dir="/home/changbae/fmri_project/MDMR/result/correlation_matrix"
output_csv_file = '/home/changbae/fmri_project/MDMR/data/predictor_variable.csv'

generate_predictor_file(source_dir, output_csv_file)
