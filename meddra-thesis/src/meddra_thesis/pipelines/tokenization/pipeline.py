# """
# This is a boilerplate pipeline 'data_abb'
# generated using Kedro 0.18.3
# """
# import pandas as pd
# import numpy as np
# from kedro.pipeline import Pipeline, node, pipeline
# import random

# from functools import partial
# from .nodes import  tokenize_data


# def create_tok_pipeline(**kwargs) -> Pipeline:
#     return Pipeline(
#         [
#             node(
#                 func=tokenize_data,
#                 inputs={
#                     "df_train": "train_post_data_text_pre",
#                     "df_test": "test_post_data_text_pre",
#                     "config" :"params:project_init"

#                 },
#                 outputs=["train_dataset", "test_dataset"],
#                 name="tokenizer",
#             )
#         ]
#     )
