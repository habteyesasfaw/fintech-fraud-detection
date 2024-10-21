import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import socket
import struct
import matplotlib.pyplot as plt
import seaborn as sns

# Handle Missing Values
def handle_missing_values(df):
    df = df.dropna()
    return df

# Data Cleaning
def data_cleaning(df):
    df = df.drop_duplicates()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

# Convert IP addresses to integer format
def ip_to_int(ip_address):
    """Convert an IP address (string or float) to an integer."""
    try:
        # If IP address is a float, convert to int (rounding if needed)
        if isinstance(ip_address, float):
            ip_address = int(ip_address)
        
        # If IP address is now an integer, convert it to IP string format
        if isinstance(ip_address, int):
            ip_address = socket.inet_ntoa(struct.pack('!I', ip_address))
        
        # Split and convert IP address (string) to integer
        octets = list(map(int, ip_address.split('.')))
        return sum([octet << (8 * i) for i, octet in enumerate(reversed(octets))])
    
    except (AttributeError, ValueError, OSError) as e:
        raise ValueError(f"Invalid IP address: {ip_address}") from e

# Merging Datasets
def merge_datasets(fraud_df, ip_df):
    """Merge fraud_df with ip_df using IP address ranges."""
    
    # Convert fraud_df IP addresses to integer format
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Ensure lower and upper bounds are integers in ip_df
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address_int'] = ip_df['upper_bound_ip_address'].astype(int)
    
    # Use a list to store matching rows
    matched_rows = []

    # Iterate over each row in the fraud dataframe
    for index, row in fraud_df.iterrows():
        ip_int = row['ip_address_int']
        # Check if this IP falls within any of the ranges in ip_df
        match = ip_df[(ip_df['lower_bound_ip_address_int'] <= ip_int) & 
                       (ip_df['upper_bound_ip_address_int'] >= ip_int)]
        if not match.empty:
            # Append matched row along with relevant IP data
            for _, ip_row in match.iterrows():
                matched_rows.append({**row, **ip_row.to_dict()})

    # Create a new DataFrame from the matched rows
    merged_df = pd.DataFrame(matched_rows)
    
    return merged_df



# Exploratory Data Analysis (EDA)
def eda(df):
    # Univariate Analysis
    print("Univariate Analysis:")
    print(df.describe())
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['purchase_value'], bins=30, kde=True)
    plt.title("Distribution of Purchase Value")
    plt.show()

    # Bivariate Analysis
    print("\nBivariate Analysis:")
    print(pd.crosstab(df['browser'], df['class'], normalize='index'))

    # Plot browser vs. class
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='browser', hue='class')
    plt.title("Fraud Cases by Browser")
    plt.show()

# Feature Engineering
def feature_engineering(df):
    df['signup_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df

# Normalization and Scaling
def normalize_and_scale(df, columns_to_scale):
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

# Encode Categorical Features
def encode_categorical(df, columns_to_encode):
    le = LabelEncoder()
    for col in columns_to_encode:
        df[col] = le.fit_transform(df[col])
    return df

# Save the processed data
def save_processed_data(df, file_name):
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")
