import numpy as np

def compute_first_order_variations(path):
    increments = np.diff(path)
    S_N = np.mean(increments**2)
    return S_N
    
def compute_second_order_variations(path):
    increments_2=-path[2:]+2*path[1:-1]-path[:-2]
    S_N = np.mean(increments_2**2)
    return S_N
    
def estimate_hurst_direct(path):

    N = len(path) - 1
    S_N = compute_first_order_variations(path)
    H_estimated_direct = -np.log(S_N)/(2*np.log(N))
    return H_estimated_direct

def estimate_hurst_direct2(path):
    n=len(path)-1
    S_N = compute_second_order_variations(path)
    H_grid=np.linespace(1e-4, 1-1e-4,10000)
    Theoretical_S=(n**(-2*H_grid))*(4-2**(2*H_grid))
    return H_grid[np.argmin(np.abs(Theoretical_S-S_N))]

