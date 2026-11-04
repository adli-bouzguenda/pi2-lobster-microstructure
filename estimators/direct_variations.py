import numpy as np

def compute_first_order_variations(path):
    increments = np.diff(path)
    S_N = np.mean(increments**2)
    return S_N

def estimate_hurst_direct(path):

    N = len(path) - 1
    S_N = compute_first_order_variations(path)
    H_estimated_direct = -np.log(S_N)/(2*np.log(N))
    return H_estimated_direct

