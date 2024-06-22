import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('generalized_trip_data.csv')

# Select quasi-identifiers
quasi_identifiers = [' pickup_longitude', ' pickup_latitude', ' dropoff_longitude', ' dropoff_latitude']

# Grouping and counting records by quasi-identifiers before anonymization
grouped_original = data.groupby(quasi_identifiers).size().reset_index(name='group_size_original')

# Analyze group sizes before anonymization
group_sizes_original = grouped_original['group_size_original']

# Print descriptive statistics of group sizes before anonymization
print("Descriptive Statistics of Group Sizes (Before Anonymization):")
print(group_sizes_original.describe())

# Determine an appropriate K value based on original data (for reference)
K_original = int(group_sizes_original.quantile(0.75))  # 75th percentile of group sizes before anonymization
print(f"Suggested K value based on 75th percentile (Before Anonymization): {K_original}")

# Applying K-Anonymity with K=8
grouped_k_anonymity = data.groupby(quasi_identifiers).filter(lambda x: len(x) >= 8)

# Grouping and counting records by quasi-identifiers after anonymization
grouped_anonymized = grouped_k_anonymity.groupby(quasi_identifiers).size().reset_index(name='group_size_anonymized')

# Analyze group sizes after anonymization
group_sizes_anonymized = grouped_anonymized['group_size_anonymized']

# Print descriptive statistics of group sizes after anonymization
print("\nDescriptive Statistics of Group Sizes after K-Anonymity (K=8):")
print(group_sizes_anonymized.describe())

# Plotting combined histograms for comparison
plt.figure(figsize=(16, 8))

# Plot histogram of group sizes before anonymization
plt.subplot(1, 2, 1)
plt.hist(group_sizes_original, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Group Size')
plt.ylabel('Frequency')
plt.title('Distribution of Group Sizes Before Anonymization')
plt.grid(True)

# Plot histogram of group sizes after anonymization
plt.subplot(1, 2, 2)
plt.hist(group_sizes_anonymized, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Group Size')
plt.ylabel('Frequency')
plt.title('Distribution of Group Sizes After K-Anonymity (K=8)')
plt.grid(True)

plt.tight_layout()
plt.show()