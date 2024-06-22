import pandas as pd

# Load the dataset
data = pd.read_csv('trip_data_10_20percent.csv')

# List of columns to drop
columns_to_drop = [' hack_license', ' vendor_id', ' rate_code', ' store_and_fwd_flag']

# Drop the columns from the dataframe
data.drop(columns_to_drop, axis=1, inplace=True)

# Save the updated dataframe as a new CSV file
new_file_path = 'trip_data_cleaned.csv'
data.to_csv(new_file_path, index=False)

print(f"New CSV file saved successfully: {new_file_path}")