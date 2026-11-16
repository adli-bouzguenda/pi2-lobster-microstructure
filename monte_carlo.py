import numpy as np
from fbm import FBM
from estimators.discrete_variations import estimate_hurst
from scipy import stats
import matplotlib.pyplot as plt
from estimators.direct_variations import estimate_hurst_direct,estimate_hurst_direct2

# MONTE CARLO PARAMETERS

H_values = [0.3, 0.5, 0.7]
N_values = [256, 1024, 4096]
M = 100

np.random.seed(42)


def compute_metrics(estimates, H_true, M):
    """Calcule biais, variance, RMSE, t-stat et p-value pour une liste d'estimations."""
    estimates = np.array(estimates)
    mean_H = np.mean(estimates)
    bias = mean_H - H_true
    var = np.var(estimates, ddof=1)
    rmse = np.sqrt(np.mean((estimates - H_true) ** 2))
    standard_error = np.sqrt(var / M)
    t_stat = bias / standard_error
    p_value = 2 * stats.t.sf(abs(t_stat), df=M - 1)
    return bias, var, rmse, t_stat, p_value


# MONTE CARLO EXPERIMENT

def monte_carlo_experiment(H, N, M):

    est_multi_1 = []
    est_multi_2 = []
    est_direct_1 = []
    est_direct_2 = []

    # 1. SIMULATION AND HURST ESTIMATION

    for i in range(M):
        process = FBM(n=N, hurst=H, length=1.0, method="daviesharte")
        path = process.fbm()

        # MULTISCALE ESTIMATORS
        H_m1, _ = estimate_hurst(path, 10, "filter1")
        est_multi_1.append(H_m1)

        H_m2, _ = estimate_hurst(path, 10, "filter2")
        est_multi_2.append(H_m2)

        # DIRECT ESTIMATORS
        H_d1 = estimate_hurst_direct(path)
        est_direct_1.append(H_d1)

        H_d2 = estimate_hurst_direct2(path)
        est_direct_2.append(H_d2)

    # 2. METRICS FOR ALL 4 ESTIMATORS

    return {
        "Multiscale (-1,1)": compute_metrics(est_multi_1, H, M),
        "Multiscale (-1,2,-1)": compute_metrics(est_multi_2, H, M),
        "Direct (-1,1)": compute_metrics(est_direct_1, H, M),
        "Direct (-1,2,-1)": compute_metrics(est_direct_2, H, M),
    }


# MONTE CARLO STUDY - COMPARISON OF 4 HURST ESTIMATORS

print("\n" + "=" * 115)
print("MONTE CARLO STUDY - COMPARISON OF 4 HURST ESTIMATORS")
print(f"Number of simulations per configuration: M = {M}")
print("=" * 115)

print(
    f"{'N':>6} | {'H':>5} | {'Method':<22} | {'Bias':>10} | "
    f"{'Variance':>12} | {'RMSE':>10} | {'t-stat':>9} | {'p-value':>9}"
)
print("-" * 115)

for N in N_values:
    for H in H_values:
        results = monte_carlo_experiment(H, N, M)

        for method_name, (bias, var, rmse, t_stat, p_value) in results.items():
            print(
                f"{N:>6} | {H:>5.2f} | {method_name:<22} | "
                f"{bias:>10.4f} | {var:>12.6f} | {rmse:>10.4f} | "
                f"{t_stat:>9.4f} | {p_value:>9.4f}"
            )

        print("-" * 115)


# CONVERGENCE STUDY - H = 0.7

rmse_dict = {
    "Multiscale (-1,1)": [],
    "Multiscale (-1,2,-1)": [],
    "Direct (-1,1)": [],
    "Direct (-1,2,-1)": [],
}

for N in N_values:
    results = monte_carlo_experiment(0.7, N, M)
    for method_name in rmse_dict:
        # L'indice 2 correspond au RMSE dans (bias, var, rmse, t_stat, p_value)
        rmse_dict[method_name].append(results[method_name][2])


# CONVERGENCE RESULTS

print("\n" + "=" * 95)
print("CONVERGENCE STUDY - H = 0.7 (RMSE)")
print("=" * 95)

print(
    f"{'N':>6} | {'Multi (-1,1)':>18} | {'Multi (-1,2,-1)':>18} | "
    f"{'Direct (-1,1)':>18} | {'Direct (-1,2,-1)':>18}"
)
print("-" * 95)

for i, N in enumerate(N_values):
    print(
        f"{N:>6} | "
        f"{rmse_dict['Multiscale (-1,1)'][i]:>18.6f} | "
        f"{rmse_dict['Multiscale (-1,2,-1)'][i]:>18.6f} | "
        f"{rmse_dict['Direct (-1,1)'][i]:>18.6f} | "
        f"{rmse_dict['Direct (-1,2,-1)'][i]:>18.6f}"
    )

print("-" * 95)

log_N = np.log(N_values)
slopes = {}

for method_name, rmses in rmse_dict.items():
    slope, intercept = np.polyfit(log_N, np.log(rmses), 1)
    slopes[method_name] = (slope, intercept)
    print(f"{method_name:<22} convergence slope: {slope:.4f}")

print("=" * 95)


# CONVERGENCE PLOT

plt.figure(figsize=(10, 6))

styles = {
    "Multiscale (-1,1)": ("royalblue", "o"),
    "Multiscale (-1,2,-1)": ("seagreen", "^"),
    "Direct (-1,1)": ("darkorange", "s"),
    "Direct (-1,2,-1)": ("crimson", "D"),
}

for method_name, rmses in rmse_dict.items():
    color, marker = styles[method_name]
    slope, intercept = slopes[method_name]
    log_rmse = np.log(rmses)
    fitted_line = slope * log_N + intercept

    plt.scatter(log_N, log_rmse, color=color, marker=marker, s=75, label=f"{method_name}")
    plt.plot(
        log_N, fitted_line, color=color, linestyle="--",
        label=f"{method_name} fit (slope = {slope:.4f})"
    )

plt.xlabel("log(N)")
plt.ylabel("log(RMSE)")
plt.title("Convergence of the 4 Hurst Estimators (H = 0.7, M = 100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
