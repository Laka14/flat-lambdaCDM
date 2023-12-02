import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate mock simulations for Ωm
num_simulations = 3000
mean_omega_m = 0.3  # Example mean value for Ωm
std_dev_omega_m = 0.05  # Example standard deviation for Ωm
mock_simulations = np.random.normal(loc=mean_omega_m, scale=std_dev_omega_m, size=num_simulations)

# Best-fit value from SN data for the specified redshift bin
best_fit_omega_m = 0.29  # Example best-fit value for Ωm from SN data

# Calculate percentiles for the mock simulations
percentiles = np.percentile(mock_simulations, [2.3, 15.9, 84.1, 97.7])

# Plotting
plt.figure(figsize=(8, 6))

# Histogram of mock simulations
plt.hist(mock_simulations, bins=30, density=True, alpha=0.7, color='gray', label='Mock Simulations')

# Vertical line for the best-fit value from SN data
plt.axvline(x=best_fit_omega_m, color='black', linestyle='-', linewidth=2, label='Best Fit (SN Data)')

# Percentile lines
for percentile_value, linestyle in zip(percentiles, ['--', ':', ':', '--']):
    plt.axvline(x=percentile_value, color='red', linestyle=linestyle, linewidth=2, label=f'{percentile_value:.1f}% Confidence Interval')

# Plot labels and legend
plt.title('Comparison of Mock Simulations with Best Fit (SN Data)')
plt.xlabel('Ωm Values')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()
