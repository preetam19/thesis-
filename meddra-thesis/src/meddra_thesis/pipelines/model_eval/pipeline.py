"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from ..model_training.nodes import create_extended_model, calculate_weights, create_base_model
from ..model_training.tokenizers import prepare_meddra_dataset, create_dataloader
from .nodes import load_model_weights, eval_loop

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [ 
        node(
                func=calculate_weights,
                inputs=["train_post_data_text_pre", "params:training_parameters"],
                outputs={"soc_weights": "soc_weights", 
                        "pt_weights": "pt_weights", 
                        "llt_weights": "llt_weights"},
                name="calculate_weights_node"
            ),
            node(
                func=create_base_model, 
                inputs=["params:soc_model", "soc_weights"],
                outputs="soc_model_initial",
                name="create_soc_model_node"
            ),
            node(
                func=prepare_meddra_dataset,
                inputs=["test_post_data_text_pre", "params:training_parameters"],  
                outputs=["meddra_dataset_test", "len_array"],
                name="prepare_meddra_dataset_node"
                ),
            node(
                func=create_dataloader,
                inputs=["meddra_dataset_test", "params:training_parameters"], 
                outputs="data_loader_test",
                name="create_dataloader_node"
            ),
            node(
                func=create_extended_model,
                inputs=["params:pt_model", "pt_weights", "soc_weights"],
                outputs="pt_model_initial",
                name="create_pt_model_node"
            ),
            node(
                func=create_extended_model,
                inputs=["params:llt_model", "llt_weights", "pt_weights", "soc_weights"],
                outputs="llt_model_initial",
                name="create_llt_model_node"
            ),
            node(
                func=load_model_weights,
                inputs=["params:training_parameters", "soc_model_initial", "pt_model_initial", "llt_model_initial"],
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