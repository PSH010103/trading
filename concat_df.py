import pandas as pd
import glob
import os

# Specify the directory containing the CSV files
csv_dir = './processed_data/'  # replace with your directory path
output_train = 'train_X.csv'
output_valid = 'valid_X.csv'
output_test = 'test_X.csv'
# Initialize lists to hold each part of the data
train_data = []
valid_data = []
test_data = []

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*_X.csv'))

# Loop through each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    
    # Determine split indices for 70%, 15%, 15%
    train_end = int(len(df) * 0.7)
    valid_end = train_end + int(len(df) * 0.15)
    
    # Split the data and append to the respective lists
    train_data.append(df.iloc[:train_end])
    valid_data.append(df.iloc[train_end:valid_end])
    test_data.append(df.iloc[valid_end:])

# Concatenate all data for each split
train_df = pd.concat(train_data, ignore_index=True)
valid_df = pd.concat(valid_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

# Save each split to a new CSV file
train_df.to_csv(output_train, index=False)
valid_df.to_csv(output_valid, index=False)
test_df.to_csv(output_test, index=False)




csv_dir = './processed_data/'  # replace with your directory path
output_train = 'train_y.csv'
output_valid = 'valid_y.csv'
output_test = 'test_y.csv'
# Initialize lists to hold each part of the data
train_data = []
valid_data = []
test_data = []

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*_y.csv'))

# Loop through each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    
    # Determine split indices for 70%, 15%, 15%
    train_end = int(len(df) * 0.7)
    valid_end = train_end + int(len(df) * 0.15)
    
    # Split the data and append to the respective lists
    train_data.append(df.iloc[:train_end])
    valid_data.append(df.iloc[train_end:valid_end])
    test_data.append(df.iloc[valid_end:])

# Concatenate all data for each split
train_df = pd.concat(train_data, ignore_index=True)
valid_df = pd.concat(valid_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

# Save each split to a new CSV file
train_df.to_csv(output_train, index=False)
valid_df.to_csv(output_valid, index=False)
test_df.to_csv(output_test, index=False)