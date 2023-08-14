import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np


def remove_special_characters(text):
    return re.sub(r"[^a-z ]", "", text)

def encode_labels(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def lowercase_text(df, column):
    df[column] = df[column].apply(str.lower)
    print(df)
    return df

def remove_special_characters_from_column(df, column):
    df[column] = df[column].apply(remove_special_characters)
    return df


def remove_stop_words(text):
    pass

