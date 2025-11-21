import os
import pandas as pd
import numpy as np

path = os.path.join('data', 'covid_data.csv')
covid_df = pd.read_csv(path, delimiter=",", dtype=str)

def adjacency_matrix_human_development_index(df: pd.DataFrame) -> np.ndarray:
    """Compute adjacency matrix based on human development index differences.
    5 if difference == 0
    4 if difference == 0.01
    3 if difference == 0.02
    2 if difference == 0.03
    1 if difference == 0.04
    0 otherwise
    Args:
        df (pd.DataFrame): DataFrame containing 'location', 'human_development
        _index', and 'continent' columns.
    Returns:
        np.ndarray: Adjacency matrix as per the defined scoring system.
    """

    tmp = df[['location', 'human_development_index', 'continent']].copy()
    tmp['human_development_index'] = pd.to_numeric(tmp['human_development_index'], errors='coerce')
    tmp = tmp[tmp['human_development_index'].notna() & tmp['continent'].notna()]

    hdi_series = tmp.groupby('location', sort=False)['human_development_index'].first()
    locations = hdi_series.index.to_list()
    hdi_values = hdi_series.to_numpy(dtype=float)

    # Pairwise absolute differences using broadcasting (very fast in numpy)
    diffs = np.abs(hdi_values[:, None] - hdi_values[None, :])
    # Round differences to 2 decimals to match original binning logic
    diffs = np.round(diffs, 2)

    # Vectorized mapping: 0->5, 0.01->4, 0.02->3, 0.03->2, 0.04->1, else 0
    targets = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    scores = np.array([5, 4, 3, 2, 1], dtype=np.int8)

    # Use broadcasting with isclose for robustness against float errors
    matrix = np.zeros_like(diffs, dtype=np.int8)
    for tval, score in zip(targets, scores):
        mask = np.isclose(diffs, tval, atol=1e-6)
        matrix[mask] = score
    
    return matrix, locations

def count_each_value(matrix: np.ndarray) -> dict:
    value_counts = {}
    unique, counts = np.unique(matrix, return_counts=True)
    for val, count in zip(unique, counts):
        value_counts[val] = count
    print(value_counts)
    return value_counts


if __name__ == "__main__":
    matrix = adjacency_matrix_human_development_index(covid_df)
    counts = count_each_value(matrix)
    print("Final counts:", counts)