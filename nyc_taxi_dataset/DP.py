import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('generalized_trip_data.csv')
print("Dataset loaded.")

# Ensure correct data types for datetime columns
print("Converting datetime columns...")
data[' pickup_datetime'] = pd.to_datetime(data[' pickup_datetime'])
data[' dropoff_datetime'] = pd.to_datetime(data[' dropoff_datetime'])
print("Datetime columns converted.")

# Select relevant columns (ignoring medallion)
data = data[[' pickup_datetime', ' dropoff_datetime', ' passenger_count', ' trip_time_in_secs',
             ' trip_distance', ' pickup_longitude', ' pickup_latitude', 
             ' dropoff_longitude', ' dropoff_latitude']]
print("Selected relevant columns.")

# Helper function to add Laplace noise
def add_laplace_noise(data, epsilon):
    sensitivity = 1  # Assumed sensitivity for count queries
    scale = sensitivity / epsilon
    noisy_data = data + np.random.laplace(0, scale, data.shape)
    return noisy_data

# Helper function to apply randomized response for Local DP
def apply_randomized_response(data, epsilon):
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    noisy_data = data.copy()
    
    for col in data.columns:
        noisy_data[col] = data[col].apply(lambda x: x if np.random.rand() < p else np.random.choice(data[col]))
    
    return noisy_data

# Helper function to add Gaussian noise
def add_gaussian_noise(data, epsilon, delta):
    sensitivity = 1  # Assumed sensitivity for count queries
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noisy_data = data + np.random.normal(0, sigma, data.shape)
    return noisy_data

# Apply Epsilon-Differential Privacy
epsilon = 0.5  # Assumed epsilon value
print("Applying Laplace noise for Epsilon-DP...")
start_time = time.time()
data_laplace = data.copy()
data_laplace[[' trip_distance', ' trip_time_in_secs']] = add_laplace_noise(data[[' trip_distance', ' trip_time_in_secs']], epsilon)
print(f"Laplace noise applied. Time taken: {time.time() - start_time} seconds.")

# Apply Local Differential Privacy
print("Applying Randomized Response for Local DP...")
start_time = time.time()
data_local_dp = data[[' trip_distance', ' trip_time_in_secs']].copy()
data_local_dp = apply_randomized_response(data_local_dp, epsilon)
print(f"Randomized Response applied. Time taken: {time.time() - start_time} seconds.")

# Apply Renyi Differential Privacy
delta = 1e-5  # Example delta value
print("Applying Gaussian noise for Rényi DP...")
start_time = time.time()
data_renyi = data.copy()
data_renyi[[' trip_distance', ' trip_time_in_secs']] = add_gaussian_noise(data[[' trip_distance', ' trip_time_in_secs']], epsilon, delta)
print(f"Gaussian noise applied. Time taken: {time.time() - start_time} seconds.")

# Visualize the original and differentially private data
print("Visualizing the results...")
plt.figure(figsize=(18, 6))

# Original data
plt.subplot(1, 4, 1)
plt.hist(data[' trip_distance'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Original Data')
plt.grid(True)

# Laplace (Epsilon-DP)
plt.subplot(1, 4, 2)
plt.hist(data_laplace[' trip_distance'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Epsilon-DP (Laplace)')
plt.grid(True)

# Local DP
plt.subplot(1, 4, 3)
plt.hist(data_local_dp[' trip_distance'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Local DP (Randomized Response)')
plt.grid(True)

# Renyi DP (Gaussian)
plt.subplot(1, 4, 4)
plt.hist(data_renyi[' trip_distance'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Trip Distance')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Renyi DP (Gaussian)')
plt.grid(True)

plt.tight_layout()
plt.show()
print("Visualization complete.")
