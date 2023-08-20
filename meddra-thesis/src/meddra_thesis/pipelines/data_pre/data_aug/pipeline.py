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
    """
    Create a pipeline for data augmentation and dataset splitting.

    This pipeline chains data augmentation node and dataset split node:
    1. Augments the data based on the specified configuration.
    2. Splits the augmented data into training and testing datasets.

    Returns:
    - Pipeline: Kedro pipeline for data augmentation.
    """
    return Pipeline(
        [
            node(
                func=partial(data_augment, col_name='llt_term', k=2, sample_frac=1.0),
                inputs={
                    "df": "pre_processed_data",
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