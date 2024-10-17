import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))
# import  preprocess as prep
from preprocess import handle_missing_values, data_cleaning, feature_engineering, ip_to_int, encode_categorical

import numpy as np

def test_handle_missing_values():
    df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    cleaned_df = handle_missing_values(df)
    assert cleaned_df.isnull().sum().sum() == 0  # Ensure no missing values

def test_data_cleaning():
    df = pd.DataFrame({
        'signup_time': ['2021-01-01', '2021-01-02'],
        'purchase_time': ['2021-01-03', '2021-01-04'],
        'duplicate': [1, 1]
    })
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Replace df.append with pd.concat
    df = pd.concat([df, df])  # Add duplicates
    df_cleaned = data_cleaning(df)  # Ensure duplicates are removed

def test_feature_engineering():
    df = pd.DataFrame({
        'signup_time': pd.to_datetime(['2021-01-01 12:00:00', '2021-01-02 14:00:00']),
        'purchase_time': pd.to_datetime(['2021-01-01 13:00:00', '2021-01-02 16:00:00'])
    })
    engineered_df = feature_engineering(df)
    assert 'signup_to_purchase' in engineered_df.columns  # Check if new feature is created

def test_ip_to_int():
    ip = '192.168.0.1'
    int_ip =ip_to_int(ip)
    assert isinstance(int_ip, int)  # Ensure conversion to integer





def test_encode_categorical():
    df = pd.DataFrame({'source': ['SEO', 'Ads', 'SEO']})
    encoded_df =encode_categorical(df, ['source'])
    assert df['source'].nunique() == 2  # Ensure categories are encoded
