import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to simulate data with varying quality and redshift
def simulate_data(redshift):
    np.random.seed(42)
    data_quality_factor = 1 - 0.1 * redshift  # Redshift-dependent data quality
    num_points = 1000
    errors = np.random.uniform(0.1, 1.0, num_points) * data_quality_factor
    hubble_values = np.random.normal(loc=67.36, scale=errors)
    return hubble_values

# Function to calculate parameters A and B
def calculate_parameters(hubble_values):
    omega_m = 0.315
    A = (np.mean(hubble_values)**2) * (1 - omega_m)
    B = (np.mean(hubble_values)**2) * omega_m
    return A, B

# Main analysis loop over different redshift bins
num_bins = 4
fit_results = {'A': [], 'B': []}

for bin_number in range(1, num_bins + 1):
    redshift = bin_number  # Simplified assumption that redshift correlates with bin number
    hubble_data = simulate_data(redshift)
    A, B = calculate_parameters(hubble_data)
    fit_results['A'].append(A)
    fit_results['B'].append(B)

# Plotting the distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(fit_results['A'], kde=True, label='A', color='blue')
plt.title('Distribution of Parameter A')
plt.xlabel('Parameter A Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(fit_results['B'], kde=True, label='B', color='orange')
plt.title('Distribution of Parameter B')
plt.xlabel('Parameter B Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()