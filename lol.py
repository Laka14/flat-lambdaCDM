import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
omega_m_values = np.linspace(0, 1, 100)  # Range of Omega_m values
std_dev_omega_m1 = 0.05
std_dev_omega_m2 = 0.34
std_dev_omega_m3 = 0.34
std_dev_omega_m4 = 0.07
z_ranges1 = [(0, 0.8)] 
z_ranges2 = [(0.8, 1.5)]
z_ranges3 = [(1.5, 2.3)]
z_ranges4 = [(2.3, 3.6)]  # Target redshift ranges


# Function to calculate probability
def calculate_probability1(omega_m, std_dev, z_range):      
    probability = norm.pdf(omega_m, loc=0.26, scale=std_dev)
    return probability
def calculate_probability2(omega_m, std_dev, z_range):      
    probability = norm.pdf(omega_m, loc=0.51, scale=std_dev)
    return probability
def calculate_probability3(omega_m, std_dev, z_range):      
    probability = norm.pdf(omega_m, loc=0.51, scale=std_dev)
    return probability
def calculate_probability4(omega_m, std_dev, z_range):      
    probability = norm.pdf(omega_m, loc=0.27, scale=std_dev)
    return probability
# Plotting for different redshift ranges
plt.figure(figsize=(12, 8))

for z_range in z_ranges1:
    omega_m_values_range = np.linspace(0, 1, 100)  # Adjust range if needed
    probabilities = [calculate_probability1(omega_m, std_dev_omega_m1, z_range) for omega_m in omega_m_values_range]
    plt.plot(omega_m_values_range, probabilities, label=f'Probability at {z_range[0]} < z < {z_range[1]}')
for z_range in z_ranges2:
    omega_m_values_range = np.linspace(0, 1, 100)  # Adjust range if needed
    probabilities = [calculate_probability2(omega_m, std_dev_omega_m2, z_range) for omega_m in omega_m_values_range]
    plt.plot(omega_m_values_range, probabilities, label=f'Probability at {z_range[0]} < z < {z_range[1]}')
for z_range in z_ranges3:
    omega_m_values_range = np.linspace(0, 1, 100)  # Adjust range if needed
    probabilities = [calculate_probability3(omega_m, std_dev_omega_m3, z_range) for omega_m in omega_m_values_range]
    plt.plot(omega_m_values_range, probabilities, label=f'Probability at {z_range[0]} < z < {z_range[1]}')
for z_range in z_ranges4:
    omega_m_values_range = np.linspace(0, 1, 100)  # Adjust range if needed
    probabilities = [calculate_probability4(omega_m, std_dev_omega_m4, z_range) for omega_m in omega_m_values_range]
    plt.plot(omega_m_values_range, probabilities, label=f'Probability at {z_range[0]} < z < {z_range[1]}')


plt.title('Probability vs. Omega_m for Forecast DESI H(z) Data')
plt.xlabel('Omega_m')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
