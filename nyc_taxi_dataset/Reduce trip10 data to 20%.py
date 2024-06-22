import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('trip_data_10.csv')

# Determine the number of rows to keep (20% of the original size)
sample_size = int(len(df) * 0.2)

# Randomly sample 20% of the dataset
sampled_df = df.sample(n=sample_size, random_state=1)  # You can change random_state for different sampling

# Save the sampled data to a new CSV file
sampled_df.to_csv('trip_data_10_20percent.csv', index=False)

print(f"Original dataset size: {len(df)} rows")
print(f"Sampled dataset size: {len(sampled_df)} rows")
print("Sampled data saved to 'trip_data_10_20percent.csv'")