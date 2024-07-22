import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_distance_matrix(file_path):
    # Distance matrix A 계산
    df = pd.read_csv(file_path)
    subjects = np.unique(df[["subject_code_1", "subject_code_2"]].values)
    n = len(subjects)
    # subject code를 index로 변경
    subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
    A = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = subject_to_index[row["subject_code_1"]]
        j = subject_to_index[row["subject_code_2"]]
        A[i, j] = -0.5 * float(row['distance']) ** 2
        A[j, i] = A[i, j]
    
    return A, subject_to_index

def calculate_gower_centered_matrix(A):
    n = A.shape[0]
    I = np.eye(n)
    one_vector = np.ones((n, 1))
    H = I - (1/n) * np.dot(one_vector, one_vector.T)
    G = np.dot(H, np.dot(A, H))
    return G

def load_design_matrix(file_path, subject_to_index):
    # Predictor variable file에서 design matrix X를 로드
    df = pd.read_csv(file_path)
    m = 1  # 하나의 predictor variable
    
    X = np.zeros((len(subject_to_index), m + 1))  # Intercept 추가를 위해 +1
    X[:, 0] = 1  # Intercept term
    
    for _, row in df.iterrows():
        i = subject_to_index[row["subject_code"]]
        X[i, 1] = row["fear"]
    
    return X

def calculate_f_statistic(G, X):
    n, m = X.shape
    H = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
    I = np.eye(n)
    
    num = np.trace(np.dot(H, G)) / (m - 1)
    denom = np.trace(np.dot(I - H, G)) / (n - m)
    
    F_stat = num / denom
    return F_stat

def permutation_test(G, X, num_permutations=5000):
    original_F_stat = calculate_f_statistic(G, X)
    permuted_F_stats = []

    for _ in tqdm(range(num_permutations), desc="Permutations"):
        permuted_X = np.copy(X)
        np.random.shuffle(permuted_X[:, 1])  # 라벨을 무작위로 섞음
        permuted_F_stat = calculate_f_statistic(G, permuted_X)
        permuted_F_stats.append(permuted_F_stat)
    
    permuted_F_stats = np.array(permuted_F_stats)
    p_value = np.sum(permuted_F_stats >= original_F_stat) / num_permutations

    return original_F_stat, p_value, permuted_F_stats

# Usage
distance_file_path = '/home/changbae/fmri_project/MDMR/result/distance_matrix/voxel_118.csv'
predictor_file_path = '/home/changbae/fmri_project/MDMR/data/predictor_variable.csv'

A, subject_to_index = calculate_distance_matrix(distance_file_path)
G = calculate_gower_centered_matrix(A)
X = load_design_matrix(predictor_file_path, subject_to_index)
#original_F_stat, p_value, permuted_F_stats = permutation_test(G, X)
original_F_stat = calculate_f_statistic(G, X)
print(f"Original F Statistic: {original_F_stat}")
#print(f"Permuted F Statistic: {permuted_F_stats}")
#print(f"P-value: {p_value}")
