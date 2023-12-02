import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate mock simulations for Ωm
num_simulations = 10000
mean_omega_m = 0.3
std_dev_omega_m = 0.05 
mock_simulations = np.random.normal(loc=mean_omega_m, scale=std_dev_omega_m, size=num_simulations)

# Best-fit value from OHD data for the specified redshift bin
best_fit_omega_m = 0.26 

# Calculate percentiles for the mock simulations
percentiles = np.percentile(mock_simulations, [2.3, 15.9, 84.1, 97.7])

# Plotting
plt.figure(figsize=(8, 6))
plt.hist(mock_simulations, bins=30, density=True, alpha=0.7, color='SkyBlue', label='Mock Simulations')
plt.axvline(x=best_fit_omega_m, color='black', linestyle='-', linewidth=2, label='Best Fit (OHD Data)')

for percentile_value, linestyle in zip(percentiles, ['--', ':', ':', '--']):
    plt.axvline(x=percentile_value, color='red', linestyle=linestyle, linewidth=2, label=f'{percentile_value:.1f}% Percentile')


plt.title('Comparison of Mock Simulations with Best Fit (OHD Data 1st bin)')
plt.xlabel('Ωm Values')
plt.ylabel('Density')
plt.legend()
plt.show()
