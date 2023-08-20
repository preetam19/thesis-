from kedro.pipeline import Pipeline, node
from .nodes import create_extended_model, train_single_model, calculate_weights, create_base_model,checkpoint, save_model
from functools import partial
from .tokenizers import prepare_meddra_dataset, create_dataloader
def create_pipeline(**kwargs):
    # train_partial = partial(
    # train_models,
    # models=["soc_model", "pt_model", "llt_model"],
    # label_weights=[kwargs["params:pretrained"], kwargs["params:pretrained"], kwargs["params:pretrained"]]
    # )
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
                outputs="soc_model",
                name="create_soc_model_node"
            ),
            node(
                func=prepare_meddra_dataset,
                inputs=["train_post_data_text_pre", "params:training_parameters"],  
                outputs=["meddra_dataset_train", "len_array_train"],
                name="prepare_meddra_dataset_node"
                ),
            node(
                func=create_dataloader,
                inputs=["meddra_dataset_train", "params:training_parameters"],  
                outputs="data_loader_train",
                name="create_dataloader_node"
            ),
            node(
                func=create_extended_model,
                inputs=["params:pt_model", "pt_weights", "soc_weights"],
                outputs="pt_model",
                name="create_pt_model_node"
            ),
            node(
                func=create_extended_model,
                inputs=["params:llt_model", "llt_weights", "pt_weights", "soc_weights"],
                outputs="llt_model",
                name="create_llt_model_node"
            ),
            node(
                func=train_single_model,
                inputs=["soc_model", "params:soc_model","params:training_parameters", "data_loader_train", "soc_weights"],
                outputs="trained_soc_model",
                name="train_soc_model_node"
            ),
            node(
                func=train_single_model,
                inputs=["pt_model", "params:pt_model", "params:training_parameters","data_loader_train", "pt_weights", "trained_soc_model"],
                outputs="trained_pt_model",
                name="train_pt_model_node"
            ),

            node(
                func=train_single_model,
                inputs=["llt_model", "params:llt_model", "params:training_parameters","data_loader_train", "llt_weights", "trained_soc_model", "trained_pt_model"],
                outputs="trained_llt_model",
                name="train_llt_model_node"
            ),
            node(
                func=save_model,
                inputs=["params:training_parameters","trained_soc_model", "trained_pt_model", "trained_llt_model"],
                outputs=None,
                name="save_model_none"
        
            ),
            node(
            func=checkpoint,
            inputs=["trained_soc_model", "trained_pt_model", "trained_llt_model"],
            outputs="checkpoint_output",
            name="checkpoint_node"
        )

        ]
    )
