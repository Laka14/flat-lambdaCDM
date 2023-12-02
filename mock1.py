import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate simulated Hubble parameter data
def generate_mock_data(planck_h0, planck_omega_m, forecast_errors):
    np.random.seed(42)  # For reproducibility
    mock_data = {}
    for zi, error in enumerate(forecast_errors):
        hubble_value = np.random.normal(loc=planck_h0, scale=error)
        mock_data[zi] = hubble_value
    return mock_data

# Function to calculate parameters A and B
def calculate_parameters(h0, omega_m):
    A = h0**2 * (1 - omega_m)
    B = h0**2 * omega_m
    return A, B
def fit_parameters(data, prior_mean, prior_std):
    def log_likelihood(params, data, prior_mean, prior_std):
        h0, omega_m = params
        predicted_data = np.array([data[zi] for zi in range(len(data))])
        prior_prob = norm.logpdf(omega_m, loc=prior_mean, scale=prior_std)
        likelihood = np.sum((predicted_data - h0)**2)
        return likelihood - prior_prob

    initial_guess = [67.36, 0.315]  # Starting values for H0 and Omega_m
    result = minimize(log_likelihood, initial_guess, args=(data, prior_mean, prior_std))

    return result.x
# Main simulation loop
num_simulations = 1000
prior_mean = 0.1430
prior_std = 0.0011

fit_results = {'bin1': {'A': [], 'B': []}, 'bin2': {'A': [], 'B': []}, 'bin3': {'A': [], 'B': []}, 'bin4': {'A': [], 'B': []}}

for _ in range(num_simulations):
    forecast_errors = np.random.uniform(0.1, 1.0, size=10)  # Example forecast errors
    mock_data = generate_mock_data(67.36, 0.315, forecast_errors)

    bins = {
        'bin1': [mock_data[zi] for zi in range(3)],
        'bin2': [mock_data[zi] for zi in range(3, 6)],
        'bin3': [mock_data[zi] for zi in range(6, 9)],
        'bin4': [mock_data[zi] for zi in range(9, 10)],
    }

    for bin_name, bin_data in bins.items():
        h0, omega_m = fit_parameters(bin_data, prior_mean, prior_std)
        A, B = calculate_parameters(h0, omega_m)
        fit_results[bin_name]['A'].append(A)
        fit_results[bin_name]['B'].append(B)

plt.figure(figsize=(12, 6))

for bin_name, params in fit_results.items():
    sns.histplot(params['A'], kde=True, label=f'A - {bin_name}', color='blue', alpha=0.5, orientation='horizontal')
    sns.histplot(params['B'], kde=True, label=f'B - {bin_name}', color='orange', alpha=0.5, orientation='horizontal')

plt.title('Distribution of Parameters A and B in Different Bins')
plt.xlabel('Frequency')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()

