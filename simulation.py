import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
from estimators.discrete_variations import estimate_hurst
from estimators.direct_variations import estimate_hurst_direct


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


# HURST EXPONENT ESTIMATION


# MULTISCALE ESTIMATOR
H_estimated = estimate_hurst(path, max_scale=10)

# DIRECT ESTIMATOR
H_hat_direct = estimate_hurst_direct(path)


# ESTIMATION RESULTS

print("\n" + "=" * 65)
print("FRACTIONAL BROWNIAN MOTION - HURST ESTIMATION")
print("=" * 65)

print(f"True Hurst exponent: {H:.4f}")
print(f"Number of increments: {N}")
print(f"Time horizon: {T}")
print("-" * 65)

print(f"{'Method':<15} | {'Estimated H':>15} | {'Absolute Error':>15}")
print("-" * 65)

# MULTISCALE ESTIMATOR
print(
    f"{'Multiscale':<15} | "
    f"{H_estimated:>15.4f} | "
    f"{abs(H_estimated - H):>15.4f}"
)

# DIRECT ESTIMATOR
print(
    f"{'Direct':<15} | "
    f"{H_hat_direct:>15.4f} | "
    f"{abs(H_hat_direct - H):>15.4f}"
)

print("=" * 65)

# TRAJECTORY INFORMATION

print("\nTRAJECTORY INFORMATION")
print("-" * 65)

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
