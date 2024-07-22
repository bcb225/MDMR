from typing import Any
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import dask.dataframe as dd
class StatHandler:
    def __init__(self):
        pass
    def temporal_correlation_by_voxel(self,masked_data, index):
        """
        Calculate the temporal correlation of the voxel at the given index with all other voxels.

        Parameters:
        - masked_data: 2D numpy array of shape (time_points, num_voxels)
        - index: int, the index of the voxel to compare against all others

        Returns:
        - correlation_vector: 1D numpy array of temporal correlation coefficients, excluding the given index
        """
        # Extract the time series for the voxel at the given index
        voxel_time_series = masked_data[:, index]
        
        # Calculate the correlation coefficients
        correlation_vector = np.array([
            np.corrcoef(voxel_time_series, other_voxel_time_series)[0, 1]
            for other_voxel_time_series in masked_data.T
        ])
        
        return correlation_vector
    def correlation_by_voxel(self, flattened_data, voxel_index):
        voxel_time_series = flattened_data[:, voxel_index]
        correlations = np.zeros(flattened_data.shape[1])
        for i in range(flattened_data.shape[1]):
            correlations[i] = np.corrcoef(voxel_time_series, flattened_data[:, i])[0, 1]
        return correlations
    def calculate_distance(self,corr1, corr2):
        p_corr, _ = pearsonr(corr1, corr2)
        distance = np.sqrt(2 * (1 - p_corr))
        return distance
    def pairwise_correlation(self, A, B):
        am = A - np.mean(A, axis=0, keepdims=True)
        bm = B - np.mean(B, axis=0, keepdims=True)
        return am.T @ bm /  (np.sqrt(
            np.sum(am**2, axis=0,
                keepdims=True)).T * np.sqrt(
            np.sum(bm**2, axis=0, keepdims=True)))
    def calculate_distance_custom(self, corr1, corr2):
        p_corr = self.pairwise_correlation(corr1,corr2)
        distance = np.sqrt(2 * (1 - p_corr))
        return distance
