"""
This is a boilerplate pipeline 'data_abb'
generated using Kedro 0.18.3
"""
import pandas as pd
import numpy as np
# import nltk
# nltk.download('wordnet')
# from nltk.corpus import wordnet
import random
from sklearn.model_selection import train_test_split



def shuffle_text(text):
    if len(text.split()) == 2:
        return " ".join(text.split()[::-1])
    else:
        return " ".join(np.random.permutation(text.split()))

def synonyms_replacement(text):
    words = text.split()
    for i, word in enumerate(words):
        synonyms = [syn.lemmas()[0].name() for syn in wordnet.synsets(word)]
        if synonyms:
            words[i] = np.random.choice(synonyms)
    return " ".join(words)

def crop_text(text):
    words = text.split()
    if len(words) < 2:
        return text   
    start_index = random.randint(0, len(words) - 2)
    end_index = random.randint(start_index + 1, len(words) - 1)
    
    cropped_words = words[start_index:end_index + 1]
    return ' '.join(cropped_words)


def data_augment(df, augment_config,col_name, k=1, sample_frac=1.0):
    aug_functions = []
    if augment_config['use_shuffling']:
        aug_functions.append(shuffle_text)
    if augment_config['use_synonyms_replacement']:
        aug_functions.append(synonyms_replacement)
    if augment_config['use_crop']:
        aug_functions.append(crop_text)
    
    sample = df.sample(frac=sample_frac).copy()
    def apply_augmentations(row):
        nonlocal k, aug_functions
        row_copies = [row.copy() for _ in range(k)]
        for copy in row_copies:
            for aug_fn in aug_functions:
                copy[col_name] = aug_fn(copy[col_name])
        return row_copies

    new_rows = sample.apply(apply_augmentations, axis=1)

    augmented_df = pd.DataFrame(
        [row for sublist in new_rows for row in sublist], columns=sample.columns
    )

    return augmented_df

def split_data(df, augment_config):
    train_df, test_df = train_test_split(df, test_size=augment_config['split_ratio'], random_state=42)
    return train_df, test_df

# def data_augmentation(df, k, col,sample_frac=1.0):
#     def shuffle_row(row):
#         nonlocal k
#         new_rows = []
#         for _ in range(k):
#             if len(row["llt_term"].split()) == 2:
#                 shuffled_llt_term = " ".join(row["llt_term"].split()[::-1])
#             else:
#                 shuffled_llt_term = " ".join(
#                     np.random.permutation(row["llt_term"].split())
#                 )
#             new_row = {
#                 "llt_term": shuffled_llt_term,
#                 "llt": row["llt"],
#                 "pt": row["pt"],
#                 "soc": row["soc"],
#             }
#             new_rows.append(new_row)
#         return new_rows

#     df_copy = df.copy()
#     sample = df_copy.sample(frac=sample_frac)
#     new_rows = sample.apply(shuffle_row, axis=1)
#     augmented_df = pd.DataFrame(
#         [row for sublist in new_rows for row in sublist], columns=df_copy.columns
#     )

#     return augmented_df