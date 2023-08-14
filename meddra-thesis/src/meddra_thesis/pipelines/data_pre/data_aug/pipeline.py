"""
This is a boilerplate pipeline 'data_abb'
generated using Kedro 0.18.3
"""
import pandas as pd
import numpy as np
from kedro.pipeline import Pipeline, node, pipeline
import random

from functools import partial
from .nodes import  data_augment, shuffle_text, synonyms_replacement, crop_text, split_data


def create_aug_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=partial(data_augment, col_name = 'llt_term', k =2, sample_frac =1.0),
                inputs={
                    "df": "preprocess_post_data_text_pre",
                    "augment_config": "params:augmentation",
                },
                outputs="data_aug",
                name="data_augmentation_node",
            ),
            node(
                func=split_data,
                inputs={
                    "df": "data_aug",
                    "augment_config": "params:augmentation",
                },
                outputs=["train_post_data_text_pre", "test_post_data_text_pre"],
                name="data_split_node",
            )
        ]
    )






    # return pipeline([
    #     node(
    #         func=functools.partial(data_augmentation,k=2, sample_frac=1.0),
    #         inputs={'df':'post_data'},
    #         outputs="data_aug",
    #         name="data_augmented",
    #         )
    #         ])