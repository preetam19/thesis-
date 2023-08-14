"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from meddra_thesis.pipelines.data_pre import data_aug as daug
from meddra_thesis.pipelines.data_pre import data_abb as dabb
from meddra_thesis.pipelines.data_pre import text_pre as dtxt
from meddra_thesis.pipelines.model_training import create_pipeline as model_training_pipeline
# from meddra_thesis.pipelines.tokenization import create_tok_pipeline 

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipeline_aug = daug.create_aug_pipeline()
    pipeline_dint_abb = dabb.create_data_abbreviation_pipeline()
    pipeline_txt = dtxt.create_preprocessing_pipeline()
    # pipeline_tok = create_tok_pipeline()
    # Concatenate the pipelines to make them run sequentially
    default_pipeline = pipeline_dint_abb +pipeline_txt +  pipeline_aug 

    return {
        "data_abb": pipeline_dint_abb,
        "data_aug": pipeline_aug,
        "data_txt": pipeline_txt,
        "model_training": model_training_pipeline(),
        "__default__": default_pipeline
    }
