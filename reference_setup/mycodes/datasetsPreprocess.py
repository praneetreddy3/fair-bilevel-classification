import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def preprocess_data(df, data_name=None):
    """
    Revise the sensitive attribute being {1,0} and the labels being {-1,1}.
    """
    if data_name == 'law':
        return preprocess_law(df)
    elif data_name == 'dutch':
        return preprocess_dutch(df)
    elif data_name == 'adult':
        return preprocess_adult(df)
    elif data_name == 'credit':
        return preprocess_credit(df)

def preprocess_law(df):
    """
    Preprocess the data by converting 'race' to 0 and 1 and 'pass_bar' to 1 and -1.

    Args:
    df (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Convert 'race' to 0 (Non-White) and 1 (White)
    df['race'] = df['race'].map({'White': 1, 'Non-White': 0})

    # Convert 'pass_bar' to -1 where it is 0.0 and 1 where it is 1.0
    df['pass_bar'] = df['pass_bar'].map({0.0: -1, 1.0: 1})

    return df


def preprocess_dutch(df):
    """
    Preprocess the data by converting 'sex' to 0 and 1 and 'occupation' to 1 and -1.

    Args:
    df (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Convert 'race' to 0 (female) and 1 (male)
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})

    # Convert 'pass_bar' to -1 where it is 0.0 and 1 where it is 1.0
    df['occupation'] = df['occupation'].map({0: -1, 1: 1})

    return df


def preprocess_adult(df):
    """
    Preprocess Adult: convert 'sex' to 0 (Female) and 1 (Male), and 'income' to
    -1 (<=50K) and 1 (>50K), matching the -1/1 label convention used elsewhere.
    """
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['income'] = df['income'].map({'<=50K': -1, '>50K': 1})
    return df


def preprocess_credit(df):
    """
    Preprocess Credit: convert 'SEX' to 0 (Female) and 1 (Male) (raw UCI encoding
    is 1=male, 2=female), and 'default' to -1 (no default) and 1 (default).
    """
    df['SEX'] = df['SEX'].map({1: 1, 2: 0})
    df['default'] = df['default'].map({0: -1, 1: 1})
    return df


def split_data_to_tensors(df, data_name=None):
    """
    Split the training data into features (x), sensitive attribute (s), and label (y),
    and convert them to PyTorch tensors.

    Returns:
    tuple: A tuple containing tensors for features (x), sensitive attribute (s), and label (y).
    """
    if data_name == 'law':
        sensitive_attributes = 'race'
        target = 'pass_bar'
    elif data_name == 'dutch':
        sensitive_attributes = 'sex'
        target = 'occupation'
    elif data_name == 'adult':
        sensitive_attributes = 'sex'
        target = 'income'
    elif data_name == 'credit':
        sensitive_attributes = 'SEX'
        target = 'default'

    # Extract features x (all columns except 'race' and 'pass_bar')
    x = df.drop(columns=[sensitive_attributes, target]).values

    # Extract sensitive attribute s ('race')
    s = df[sensitive_attributes].values

    # Extract label y ('pass_bar')
    y = df[target].values

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32) #
    s_tensor = torch.tensor(s, dtype=torch.float32) #
    y_tensor = torch.tensor(y, dtype=torch.float32) #

    return x_tensor, s_tensor, y_tensor

def scale_data(X_train, X_test, data_name='law', scaling_method='minmax'):
    """
    Scale the data using specified scaling method.
    
    Parameters:
    - X_Train: DataFrame, the training data.
    - X_test: DataFrame, the testing data.
    - data_name: Optional, the name of the dataset.
    - scaling_method: str, the method for scaling ('minmax', 'standard', 'robust').
    
    Returns:
    - DataFrame, the scaled data.
    """

    # Initialize the scaler based on the input scaling method
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}")

    # Determine which columns to scale
    if data_name == 'law':
        columns_to_exclude = ["race"]
    elif data_name == 'dutch':
        columns_to_exclude = ["sex"]
    elif data_name == 'adult':
        columns_to_exclude = ["sex"]
    elif data_name == 'credit':
        columns_to_exclude = ["SEX"]

    columns_to_scale = [col for col in X_train.columns if col not in columns_to_exclude]
    
    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled

def detect_outliers(df):
    outliers = {}
    for column in df.columns[:-1]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    return outliers


    
    


    
