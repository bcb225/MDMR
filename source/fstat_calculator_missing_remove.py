import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

##Participant file에서 missing 된 값이 있으면 사후에 design matrix를 만들때 제거하고 만드는 방식.
##엄밀하지 않은 방식이기 때문에, 참고로만 사용해야 함.
##나중에는 관심이 있는 variable에 missing value가 있는 사람을 미리 design matrix를 만드는 파일에서 제거하고 분석을 진행해야 함.
##아니면, design matrix에서 제거한 사람, 그리고 design matrix를 만들게 된 사람의 list와 index를 모두 기록해두어야 함
##missing value를 제거하지 않고 분석하는 기존 코드도 남겨두었음.

def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)
    subjects = np.unique(df[["subject_code_1", "subject_code_2"]].values)
    n = len(subjects)
    subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
    A = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = subject_to_index[row["subject_code_1"]]
        j = subject_to_index[row["subject_code_2"]]
        A[i, j] = -0.5 * float(row['distance']) ** 2
        A[j, i] = A[i, j]
    
    return A, subject_to_index, subjects

def calculate_gower_centered_matrix(A):
    n = A.shape[0]
    I = np.eye(n)
    one_vector = np.ones((n, 1))
    H = I - (1/n) * np.dot(one_vector, one_vector.T)
    G = np.dot(H, np.dot(A, H))
    return G

def load_design_matrix(file_path, predictor, subject_to_index):
    df = pd.read_csv(file_path)
    m = 1
    X = np.zeros((len(subject_to_index), m + 1))
    X[:, 0] = 1
    
    missing_subjects = set()
    
    for _, row in df.iterrows():
        fmri_code = row["fmri_code"]
        subject_code = f"sub-{fmri_code}"
        if subject_code in subject_to_index:
            i = subject_to_index[subject_code]
            X[i, 1] = row[predictor]
        else:
            missing_subjects.add(subject_code)
            print(f"Subject {subject_code} not in voxel distance matrix!")
    
    valid_indices = ~np.isnan(X).any(axis=1)
    X = X[valid_indices]
    
    return X, valid_indices

def filter_matrices(G, X, valid_indices):
    G_filtered = G[valid_indices][:, valid_indices]
    X_filtered = X
    return G_filtered, X_filtered

def calculate_f_statistic(G, X):
    n, m = X.shape
    H = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
    I = np.eye(n)
    
    num = np.trace(np.dot(H, G)) / (m - 1)
    denom = np.trace(np.dot(I - H, G)) / (n - m)
    
    F_stat = num / denom
    return F_stat

def permutation_test(G, X, num_permutations=15000):
    original_F_stat = calculate_f_statistic(G, X)
    permuted_F_stats = []

    for _ in tqdm(range(num_permutations), desc="Permutations"):
        permuted_X = np.copy(X)
        np.random.shuffle(permuted_X[:, 1])
        permuted_F_stat = calculate_f_statistic(G, permuted_X)
        permuted_F_stats.append(permuted_F_stat)
    
    permuted_F_stats = np.array(permuted_F_stats)
    p_value = np.sum(permuted_F_stats >= original_F_stat) / num_permutations

    return original_F_stat, p_value, permuted_F_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run permutation test on fMRI data")
    parser.add_argument("--voxel_num", type=str, required=True, help="Voxel")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor")
    args = parser.parse_args()
    
    distance_file_path = f"../toy_result/distance_matrix/voxel_{args.voxel_num}.csv"
    predictor_file_path = "../toy_result/participant_demo_clinical.csv"
    predictor = args.predictor
    
    A, subject_to_index, subjects = calculate_distance_matrix(distance_file_path)
    G = calculate_gower_centered_matrix(A)
    X, valid_indices = load_design_matrix(predictor_file_path, predictor, subject_to_index)
    
    G_filtered, X_filtered = filter_matrices(G, X, valid_indices)
    
    original_F_stat, p_value, permuted_F_stats = permutation_test(G_filtered, X_filtered)
    
    print(f"Original F Statistic: {original_F_stat}")
    print(f"P-value: {p_value}")
