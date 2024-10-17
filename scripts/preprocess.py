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
def ip_to_int(ip_address):
    try:
        if isinstance(ip_address, float):
            ip_address = int(ip_address)
        if isinstance(ip_address, int):
            ip_address = socket.inet_ntoa(struct.pack('!I', ip_address))
        octets = list(map(int, ip_address.split('.')))
        return sum([octet << (8 * i) for i, octet in enumerate(reversed(octets))])
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid IP address: {ip_address}")

# Merging function example
def merge_datasets(fraud_df, ip_df):
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
    merged_df = fraud_df.merge(ip_df, left_on='ip_address_int', right_on='lower_bound_ip_address_int', how='left')
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
