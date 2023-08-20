import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np

def remove_special_characters(text):
    """
    Remove all special characters from the given text, retaining only alphabets and spaces.
    
    Parameters:
    - text (str): Input string.
    
    Returns:
    - str: String with special characters removed.
    """
    return re.sub(r"[^a-z ]", "", text)

def encode_labels(df, columns):
    """
    Encode labels of specified columns in the dataframe using label encoding.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - columns (list): List of columns to be label encoded.
    
    Returns:
    - pd.DataFrame: Dataframe with specified columns label encoded.
    """
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def lowercase_text(df, column):
    """
    Convert the content of a specified column in the dataframe to lowercase.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - column (str): Column to be converted to lowercase.
    
    Returns:
    - pd.DataFrame: Dataframe with the specified column in lowercase.
    """
    df[column] = df[column].apply(str.lower)
    return df

def remove_special_characters_from_column(df, column):
    """
    Remove special characters from a specified column in the dataframe.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - column (str): Column from which special characters should be removed.
    
    Returns:
    - pd.DataFrame: Dataframe with special characters removed from the specified column.
    """
    df[column] = df[column].apply(remove_special_characters)
    return df

def remove_stop_words(text):
    # Dummy function used as an example in the report
    pass
