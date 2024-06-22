import pandas as pd

# Load the cleaned dataset
print("Loading dataset...")
data = pd.read_csv('trip_data_cleaned.csv')

# Function to generalize coordinates
def generalize_coordinates(coord):
    return round(coord, 2)  # Example: Generalize to two decimal places

# Apply generalization to coordinates
print("Generalizing coordinates...")
data[' pickup_longitude'] = data[' pickup_longitude'].apply(generalize_coordinates)
data[' pickup_latitude'] = data[' pickup_latitude'].apply(generalize_coordinates)
data[' dropoff_longitude'] = data[' dropoff_longitude'].apply(generalize_coordinates)
data[' dropoff_latitude'] = data[' dropoff_latitude'].apply(generalize_coordinates)

# Save generalized data to CSV
print("Saving generalized data...")
data.to_csv('generalized_trip_data.csv', index=False)

# Print summary
print("Data generalized and saved to 'generalized_trip_data.csv'")