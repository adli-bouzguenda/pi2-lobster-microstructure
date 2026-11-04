import numpy as np
from fbm import FBM
from estimators.discrete_variations import estimate_hurst
from scipy import stats
import matplotlib.pyplot as plt
from estimators.direct_variations import estimate_hurst_direct

# MONTE CARLO PARAMETERS

H_values = [0.3, 0.5, 0.7]
N_values = [256, 1024, 4096]
M = 100

np.random.seed(42)

# MONTE CARLO EXPERIMENT

def monte_carlo_experiment(H, N, M):

    H_estimates = []
    H_estimate_direct = []

    # 1. SIMULATION AND HURST ESTIMATION

    for i in range(M):

        process = FBM(n=N, hurst=H, length=1.0, method="daviesharte")
        path = process.fbm()

        # MULTISCALE ESTIMATOR
        H_hat = estimate_hurst(path, max_scale=10)
        H_estimates.append(H_hat)

        # DIRECT ESTIMATOR
        H_hat_direct = estimate_hurst_direct(path)
        H_estimate_direct.append(H_hat_direct)

    # 2. BIAS AND VARIANCE

    # MULTISCALE ESTIMATOR
    mean_H = np.mean(H_estimates)
    bias = mean_H - H
    var = np.var(H_estimates, ddof=1)

    # DIRECT ESTIMATOR
    mean_H_direct = np.mean(H_estimate_direct)
    bias_direct = mean_H_direct - H
    var_direct = np.var(H_estimate_direct, ddof=1)

    # 3. RMSE

    H_estimates = np.array(H_estimates)
    H_estimate_direct = np.array(H_estimate_direct)

    # MULTISCALE ESTIMATOR
    squared_errors = (H_estimates - H)**2
    rmse = np.sqrt(np.mean(squared_errors))

    # DIRECT ESTIMATOR
    squared_errors_direct = (H_estimate_direct - H)**2
    rmse_direct = np.sqrt(np.mean(squared_errors_direct))

    # 4. STUDENT T-TEST

    # MULTISCALE ESTIMATOR
    standard_error = np.sqrt(var / M)
    t_stat = bias / standard_error
    p_value = 2 * stats.t.sf(abs(t_stat), df=M - 1)

    # DIRECT ESTIMATOR
    standard_error_direct = np.sqrt(var_direct / M)
    t_stat_direct = bias_direct / standard_error_direct
    p_value_direct = 2 * stats.t.sf(abs(t_stat_direct), df=M - 1)

    # 5. RETURN RESULTS

    return (
        bias, var, rmse, t_stat, p_value,
        bias_direct, var_direct, rmse_direct,
        t_stat_direct, p_value_direct
    )


# MONTE CARLO STUDY - COMPARISON OF HURST ESTIMATORS

print("\n" + "=" * 105)
print("MONTE CARLO STUDY - MULTISCALE VS DIRECT ESTIMATOR")
print(f"Number of simulations per configuration: M = {M}")
print("=" * 105)

print(
    f"{'N':>6} | {'H':>5} | {'Method':>10} | {'Bias':>10} | "
    f"{'Variance':>12} | {'RMSE':>10} | {'t-stat':>9} | {'p-value':>9}"
)
print("-" * 105)

for N in N_values:
    for H in H_values:

        (bias, var, rmse, t_stat, p_value,
         bias_direct, var_direct, rmse_direct,
         t_stat_direct, p_value_direct) = monte_carlo_experiment(H, N, M)

        # Multiscale estimator
        print(
            f"{N:>6} | {H:>5.2f} | {'Multiscale':>10} | "
            f"{bias:>10.4f} | {var:>12.6f} | {rmse:>10.4f} | "
            f"{t_stat:>9.4f} | {p_value:>9.4f}"
        )

        # Direct estimator
        print(
            f"{N:>6} | {H:>5.2f} | {'Direct':>10} | "
            f"{bias_direct:>10.4f} | {var_direct:>12.6f} | "
            f"{rmse_direct:>10.4f} | {t_stat_direct:>9.4f} | "
            f"{p_value_direct:>9.4f}"
        )

        print("-" * 105)


# CONVERGENCE STUDY - H = 0.7


rmse_values = []
rmse_values_direct = []

for N in N_values:

    (bias, var, rmse, t_stat, p_value,
     bias_direct, var_direct, rmse_direct,
     t_stat_direct, p_value_direct) = monte_carlo_experiment(0.7, N, M)

    rmse_values.append(rmse)
    rmse_values_direct.append(rmse_direct)


# Multiscale estimator
log_N = np.log(N_values)
log_rmse = np.log(rmse_values)

slope, intercep = np.polyfit(log_N, log_rmse, 1)
fitted_line = slope * log_N + intercep


# Direct estimator
log_rmse_direct = np.log(rmse_values_direct)

slope_direct, intercep_direct = np.polyfit(log_N, log_rmse_direct, 1)
fitted_line_direct = slope_direct * log_N + intercep_direct


# CONVERGENCE RESULTS

print("\n" + "=" * 65)
print("CONVERGENCE STUDY - H = 0.7")
print("=" * 65)

print(f"{'N':>8} | {'RMSE Multiscale':>18} | {'RMSE Direct':>18}")
print("-" * 65)

for i, N in enumerate(N_values):
    print(
        f"{N:>8} | {rmse_values[i]:>18.6f} | "
        f"{rmse_values_direct[i]:>18.6f}"
    )

print("-" * 65)
print(f"Multiscale convergence slope: {slope:.4f}")
print(f"Direct convergence slope:     {slope_direct:.4f}")
print("=" * 65)

# CONVERGENCE PLOT

plt.figure(figsize=(9, 6))

# Multiscale estimator
plt.scatter(
    log_N, log_rmse,
    color="royalblue",
    marker="o",
    s=75,
    label="Multiscale - Monte Carlo"
)

plt.plot(
    log_N, fitted_line,
    color="royalblue",
    linestyle="--",
    label=f"Multiscale regression (slope = {slope:.4f})"
)

# Direct estimator
plt.scatter(
    log_N, log_rmse_direct,
    color="darkorange",
    marker="s",
    s=75,
    label="Direct - Monte Carlo"
)

plt.plot(
    log_N, fitted_line_direct,
    color="darkorange",
    linestyle="--",
    label=f"Direct regression (slope = {slope_direct:.4f})"
)

plt.xlabel("log(N)")
plt.ylabel("log(RMSE)")
plt.title("Convergence of Hurst Estimators (H = 0.7, M = 100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
