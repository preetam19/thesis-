"""
This is a boilerplate pipeline 'data_abb'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_abb
from functools import partial
def create_data_abbreviation_pipeline(**kwargs) -> Pipeline:
    """
    Create a pipeline for the data abbreviation process.

    This pipeline uses the data abbreviation logic defined in the `data_abb` function
    to transform the `merged_raw_data` into its abbreviated form, guided by the parameters
    specified in 'params:data_abbreviation'.

    Parameters:
    - **kwargs: Additional keyword arguments (not currently used in this function).

    Returns:
    - Pipeline: Kedro pipeline for data abbreviation.
    """
    
    return Pipeline(
        [
            node(
                func=data_abb, 
                inputs=['merged_raw_data', 'params:data_abbreviation'],  
                outputs="abbreviated_data", 
                name="data_abbreviation_node",  
            ),
        ]
    )