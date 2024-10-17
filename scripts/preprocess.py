import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import socket
import struct

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

# Exploratory Data Analysis (EDA)
def eda(df):
    print("Univariate Analysis:")
    print(df.describe())
    print("\nBivariate Analysis:")
    print(pd.crosstab(df['browser'], df['class'], normalize='index'))

# Convert IP addresses to integer format


# Function to convert IP addresses to integer
def ip_to_int(ip_address):
    """Convert an IP address (string or float) to an integer."""
    try:
        # If ip_address is a float, convert to int (rounding if needed)
        if isinstance(ip_address, float):
            ip_address = int(ip_address)
        
        # If ip_address is now an integer, convert it to IP string format
        if isinstance(ip_address, int):
            # Convert integer to IP format, and then back to int to ensure validity
            ip_address = socket.inet_ntoa(struct.pack('!I', ip_address))
        
        # Split and convert IP address (string) to integer
        octets = list(map(int, ip_address.split('.')))
        return sum([octet << (8 * i) for i, octet in enumerate(reversed(octets))])
    
    except (AttributeError, ValueError, OSError) as e:
        raise ValueError(f"Invalid IP address: {ip_address}") from e

# Merging function
def merge_datasets(fraud_df, ip_df):
    """Merge fraud_df with ip_df using IP address ranges."""
    
    # Convert fraud_df IP addresses to integer format
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Ensure lower and upper bounds are integers in ip_df
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address_int'] = ip_df['upper_bound_ip_address'].astype(int)
    
    # Perform merge with conditions
    merged_df = fraud_df.merge(ip_df, how='left', left_on='ip_address_int', right_on='lower_bound_ip_address_int')

    # Filter out rows where the IP address is not within the bounds
    merged_df = merged_df[(merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) &
                          (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int'])]
    
    return merged_df

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
