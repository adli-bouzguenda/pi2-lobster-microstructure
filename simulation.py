import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
from estimators.discrete_variations import estimate_hurst
from estimators.direct_variations import estimate_hurst_direct, estimate_hurst_direct2


# SIMULATION PARAMETERS

H = 0.7
N = 1024
T = 1.0

np.random.seed(42)


# FRACTIONAL BROWNIAN MOTION SIMULATION

process = FBM(
    n=N,
    hurst=H,
    length=T,
    method="daviesharte"
)

t = process.times()
path = process.fbm()


# HURST EXPONENT ESTIMATION (4 ESTIMATORS)

# MULTISCALE ESTIMATORS (C unknown)
H_multi_1, C_est_1 = estimate_hurst(path, 10, "filter1")
H_multi_2, C_est_2 = estimate_hurst(path, 10, "filter2")

# DIRECT ESTIMATORS (C = 1 known)
H_direct_1 = estimate_hurst_direct(path)
H_direct_2 = estimate_hurst_direct2(path)

# ESTIMATION RESULTS

print("\n" + "=" * 80)
print("FRACTIONAL BROWNIAN MOTION - HURST ESTIMATION")
print("=" * 80)

print(f"True Hurst exponent: {H:.4f}")
print(f"True Scale parameter C: 1.0000")
print(f"Number of increments: {N}")
print(f"Time horizon: {T}")
print("-" * 80)

print(f"{'Method':<22} | {'Estimated H':>13} | {'Abs Error (H)':>15} | {'Estimated C':>13}")
print("-" * 80)

# MULTISCALE ESTIMATORS
print(
    f"{'Multiscale (-1,1)':<22} | "
    f"{H_multi_1:>13.4f} | "
    f"{abs(H_multi_1 - H):>15.4f} | "
    f"{C_est_1:>13.4f}"
)

print(
    f"{'Multiscale (-1,2,-1)':<22} | "
    f"{H_multi_2:>13.4f} | "
    f"{abs(H_multi_2 - H):>15.4f} | "
    f"{C_est_2:>13.4f}"
)

# DIRECT ESTIMATORS
print(
    f"{'Direct (-1,1)':<22} | "
    f"{H_direct_1:>13.4f} | "
    f"{abs(H_direct_1 - H):>15.4f} | "
    f"{'1.0000 (fixed)':>13}"
)

print(
    f"{'Direct (-1,2,-1)':<22} | "
    f"{H_direct_2:>13.4f} | "
    f"{abs(H_direct_2 - H):>15.4f} | "
    f"{'1.0000 (fixed)':>13}"
)

print("=" * 80)

# TRAJECTORY INFORMATION

print("\nTRAJECTORY INFORMATION")
print("-" * 80)

print(f"Number of time points: {len(t)}")
print(f"Number of path values: {len(path)}")
print(f"First 5 path values: {path[:5]}")

# FRACTIONAL BROWNIAN MOTION PLOT

plt.figure(figsize=(10, 5))

plt.plot(t, path, label=f"fBm (H = {H})")

plt.xlabel("Time")
plt.ylabel("B_H(t)")
plt.title(f"Fractional Brownian Motion (H = {H}, N = {N})")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
