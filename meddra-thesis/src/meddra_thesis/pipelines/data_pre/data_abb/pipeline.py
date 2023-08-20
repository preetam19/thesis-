"""
This is a boilerplate pipeline 'data_abb'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_abb
from functools import partial
def create_data_abbreviation_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=data_abb,
                inputs= ['merged_raw_data' , 'params:data_abbreviation'],
                outputs="abbreviated_data",
                name="data_abbreviation_node",
            ),
        ]
    )