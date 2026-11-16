import numpy as np

def compute_variations(path, m, filter_type):
    if filter_type=="filter1":
        return path[m:]-path[:-m]
    if filter_type=="filter2":
        return path[2*m:] - 2 *path[m:-m] + path[:-2*m]
    else:
        raise ValueError("Error")

def compute_quadratic_variations(path, m, filter_type):
    variations = compute_variations(path, m, filter_type)
    squared_variations=variations**2
    S = np.mean(squared_variations)
    return S

def compute_multiscale_variations(path, max_scale, filter_type):
    scales = np.arange(1, max_scale+1)
    S_values = []
    for m in scales :
        S = compute_quadratic_variations(path, m,filter_type)
        S_values.append(S)
    return scales, np.array(S_values)

def estimate_hurst(path, max_scale, filter_type):
    N=len(path)-1
    scales, S_values = compute_multiscale_variations(path, max_scale, filter_type)
    log_scales = np.log(scales)
    log_S = np.log(S_values)
    slope, interecept = np.polyfit(log_scales, log_S, 1)
    H_estimated = slope/2
    S_1=S_values[0]
    if filter_type=="filter1":
        C_estimated=np.sqrt(S_1*(N**(2*H_estimated)))
    elif filter_type=="filter2":
        C_estimated=np.sqrt((S_1*(N **(2*H_estimated)))/(4-2**(2*H_estimated)))
    
    return H_estimated, C_estimated

