import numpy as np

def compute_variations(path, m):
    variations = path[2*m:] - 2 *path[m:-m] + path[:-2*m]
    return variations

def compute_quadratic_variations(path, m):
    variations = compute_variations(path, m)

    squared_variations = variations ** 2

    S = np.mean(squared_variations)

    return S

def compute_multiscale_variations(path, max_scale=10):

    scales = np.arange(1, max_scale+1)

    S_values = []

    for m in scales :
        S = compute_quadratic_variations(path, m)
        S_values.append(S)

    return scales, np.array(S_values)

def estimate_hurst(path, max_scale=10):

    scales, S_values = compute_multiscale_variations(path, max_scale)

    log_scales = np.log(scales)
    log_S = np.log(S_values)
    slope, interecept = np.polyfit(log_scales, log_S, 1)
    H_estimated = slope/2

    return H_estimated

