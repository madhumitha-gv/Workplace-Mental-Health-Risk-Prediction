"""
Preprocessing utilities for Mental Health Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess mental health survey data
    """
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = df_clean.columns.str.replace('\xa0', ' ')
    
    # Ordinal mappings
    mappings = {
        'Yes': 1, 'No': 0, 'Maybe': 0.5,
        'Often': 3, 'Sometimes': 2, 'Rarely': 1, 'Never': 0,
        'Always': 3,
        'Very easy': 4, 'Somewhat easy': 3, 'Neither easy nor difficult': 2,
        'Somewhat difficult': 1, 'Very difficult': 0,
        "I don't know": 0.5, 'I am not sure': 0.5, "Don't know": 0.5,
        'Not eligible for coverage / N/A': 0,
        'Some of them': 0.5, 'Not applicable to me': 0.5,
        '1-25%': 1, '26-50%': 2, '50-75%': 3, '76-100%': 4
    }
    
    # Apply ordinal encoding
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].map(mappings)
    
    # Label encode remaining categorical
    for col in df_clean.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].fillna('Missing').astype(str))
    
    # Fill missing numerical values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    df_clean = df_clean.fillna(0)
    
    return df_clean

def create_composite_indices(df):
    """
    Create Mental Health Support, Stigma, and Openness indices
    """
    df_features = df.copy()
    
    # Mental Health Support Index
    support_cols = [
        'Does your employer provide mental health benefits as part of healthcare coverage?',
        'Do you know the options for mental health care available under your employer-provided coverage?',
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
        'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'
    ]
    
    support_data = df_features[support_cols].copy()
    for col in support_cols:
        mn, mx = support_data[col].min(), support_data[col].max()
        if mx > mn:
            support_data[col] = (support_data[col] - mn) / (mx - mn)
    
    df_features['MH_Support_Index'] = support_data.mean(axis=1)
    
    # Workplace Stigma Index
    stigma_cols = [
        'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
        'Would you feel comfortable discussing a mental health disorder with your coworkers?',
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
        'Do you feel that being identified as a person with a mental health issue would hurt your career?',
        'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?'
    ]
    
    stigma_data = df_features[stigma_cols].copy()
    for col in stigma_cols:
        mn, mx = stigma_data[col].min(), stigma_data[col].max()
        if mx > mn:
            if 'comfortable' in col.lower():
                stigma_data[col] = 1 - ((stigma_data[col] - mn) / (mx - mn))
            else:
                stigma_data[col] = (stigma_data[col] - mn) / (mx - mn)
    
    df_features['Stigma_Index'] = stigma_data.mean(axis=1)
    
    # Organizational Openness Score
    openness_cols = [
        'Do you feel that your employer takes mental health as seriously as physical health?',
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
        'Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?',
        'Would you bring up a mental health issue with a potential employer in an interview?',
        'Would you be willing to bring up a physical health issue with a potential employer in an interview?'
    ]
    
    openness_data = df_features[openness_cols].copy()
    for col in openness_cols:
        mn, mx = openness_data[col].min(), openness_data[col].max()
        if mx > mn:
            openness_data[col] = (openness_data[col] - mn) / (mx - mn)
    
    df_features['Openness_Score'] = openness_data.mean(axis=1)
    
    return df_features
