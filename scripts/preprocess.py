import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import socket
import struct
# Handle Missing Values
def handle_missing_values(df):
    # Dropping rows with missing values 
    df = df.dropna()
    return df

# Data Cleaning
def data_cleaning(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types (e.g., ensuring dates are in datetime format)
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


def ip_to_int(ip_address):
    """
    Convert an IP address (either as a string or decimal integer) to an integer.
    If the IP address is a decimal (integer or float), it attempts to convert it 
    to a standard IP address format before processing.
    """
    try:
        # If ip_address is a float, it's invalid as an IP string
        if isinstance(ip_address, float):
            # Optionally convert float to int if it looks like an integer IP
            ip_address = int(ip_address)
        
        # If ip_address is now an integer, convert it to IP string format
        if isinstance(ip_address, int):
            ip_address = socket.inet_ntoa(struct.pack('!I', ip_address))
        
        # Split and convert IP address (string) to integer
        octets = list(map(int, ip_address.split('.')))
        return sum([octet << (8 * i) for i, octet in enumerate(reversed(octets))])
    
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid IP address: {ip_address}")

# Merging function example
def merge_datasets(fraud_df, ip_df):
    """
    Merge fraud_df with ip_df using the IP address ranges after converting IPs to integers.
    """
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
    # Perform the merge (just an example, adjust to your actual logic)
    merged_df = fraud_df.merge(ip_df, left_on='ip_address_int', right_on='lower_bound_ip_address_int', how='left')
    return merged_df



# Feature Engineering
def feature_engineering(df):
    # Transaction frequency and velocity
    df['signup_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # Time-based features
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
