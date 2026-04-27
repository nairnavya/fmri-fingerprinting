import numpy as np

# ----- INCREASING TIME POINTS BY ADDING LR AND RL - BETTER ESTIMATE OF CORR[i, j] -----
def concatenate_runs(lr_timeseries, rl_timeseries): 
    if lr_timeseries.shape[1] != rl_timeseries.shape[1]:
        raise ValueError("LR and RL have different number of nodes")

    return np.vstack([lr_timeseries, rl_timeseries])
    # if LR = 405 × 268 and RL = 405 × 268, combined = 810 × 268
    

# ----- CALCULATE MATRIX OF PEARSON'S COEFFICIENTS BETWEEN NODES -----
def compute_correlation_matrix(timeseries):
    
    corr = np.corrcoef(timeseries, rowvar=False) # rowvar=False means columns are variables and rows are observations
    corr = np.clip(corr, -0.999999, 0.999999) # if value > 0.999999 → set to 0.999999 since arctanh(1) = ∞  

    return corr


# ----- MAKING CORRELATIONS MORE STABLE AND STANDARD BY USING FISHER Z-TRANSFORMATION -----
def fisher_z_transform(corr_matrix):
    z = np.arctanh(corr_matrix)
    np.fill_diagonal(z, 0)

    return z


def compute_fc_matrix(timeseries):
    corr = compute_correlation_matrix(timeseries)
    z = fisher_z_transform(corr)

    return z


# ----- SAVING ONLY UNIQUE CONNECTIONS WHERE (i < j) & CONVERTING TO A 1D VECTOR -----
def vectorize_upper_triangle(matrix):
    indices = np.triu_indices_from(matrix, k=1)

    return matrix[indices]

# matrix → vector of edges OR 268 × 268 matrix → 1D vectorsMatrix:
# e.g. 
#[ 1   a   b
#  a   1   c
#  b   c   1 ]
# is converted to 
# [a, b, c]