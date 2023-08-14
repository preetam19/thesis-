"""
This is a boilerplate pipeline 'data_abb'
generated using Kedro 0.18.3
"""

def split_words(text, max_length=4):
    words = [word for word in text.split() if len(word) < max_length]
    return words if words else None

def contains_word(words, list_to_check):
    return any(word in words for word in set(list_to_check))

def combine_unique_words(row, col1, col2):
    words1 = set(row[col1].split())
    words2 = row[col2].split()
    
    unique_words2 = [word for word in words2 if word not in words1]
    return row[col1] + ' ' + ' '.join(unique_words2)

def data_abb(df,params, max_word_length=4):
    col1, col2, unwanted_words = params['col1'], params['col2'], params['unwanted_words']
    df = df.copy()
    print(col2)
    df['words'] = df[col1].apply(lambda x: split_words(x, max_word_length))
    
    words_to_check = df[df['words'].notna()]['words'].tolist()
    words_to_check = [[word for word in sublist if word not in unwanted_words] for sublist in words_to_check]
    words_to_check = [sublist for sublist in words_to_check if sublist]
    flattened = [word for sublist in words_to_check for word in sublist]
    flattened_to_check = [word for word in flattened if word == word.upper() and not word.isdigit()]
    
    filter_mask = df[col1].apply(lambda x: contains_word(x.split(), flattened_to_check))
    filtered_df = df[filter_mask]
    filtered_df[col1] = filtered_df.apply(lambda row: combine_unique_words(row, col1, col2), axis=1)
    
    df.update(filtered_df)
    return df