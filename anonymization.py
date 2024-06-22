import pandas as pd
import random
from datetime import timedelta

# Load the dataset
data = pd.read_csv('trip_data_cleaned.csv')

# Function to round coordinates
def round_coordinates(coord):
    return round(coord, 3)

# Round latitude and longitude to 3 decimal places
data[' pickup_longitude'] = data[' pickup_longitude'].apply(round_coordinates)
data[' pickup_latitude'] = data[' pickup_latitude'].apply(round_coordinates)
data[' dropoff_longitude'] = data[' dropoff_longitude'].apply(round_coordinates)
data[' dropoff_latitude'] = data[' dropoff_latitude'].apply(round_coordinates)

data.to_csv('anonymized_trip_data.csv', index=False)

print("Anonymized dataset saved successfully.")