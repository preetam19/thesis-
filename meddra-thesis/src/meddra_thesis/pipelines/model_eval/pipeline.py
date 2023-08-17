"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from ..model_training.nodes import checkpoint,create_extended_model, calculate_weights, create_base_model
from ..model_training.tokenizers import prepare_meddra_dataset, create_dataloader
from .nodes import load_model_weights, eval_loop

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [ 



            node(
                func=prepare_meddra_dataset,
                inputs=["test_post_data_text_pre", "params:training_parameters"],  
                outputs=["meddra_dataset_test", "len_array"],
                name="prepare_meddra_dataset_node_eval"
                ),
            node(
                func=create_dataloader,
                inputs=["meddra_dataset_test", "params:training_parameters"], 
                outputs="data_loader_test",
                name="create_dataloader_node_eval"
            ),
            node(
                func=load_model_weights,
                inputs=["params:training_parameters", "soc_model", "pt_model", "llt_model","checkpoint_output"],
                outputs="trained_models",
                name="load_weights_node"
            ),
            node(
                func = eval_loop, 
                inputs=["data_loader_test","len_array", "trained_models"],
                outputs=None
                 )
            ] 
            )