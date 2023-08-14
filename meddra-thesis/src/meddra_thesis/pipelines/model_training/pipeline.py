from kedro.pipeline import Pipeline, node
from .nodes import create_extended_model, train_single_model, calculate_weights, create_base_model, create_train_dataloader
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
                inputs=["train_post_data_text_pre", "params:pretrained"],
                outputs={"soc_weights": "soc_weights", 
                        "pt_weights": "pt_weights", 
                        "llt_weights": "llt_weights"},
                name="calculate_weights_node"
            ),WWWWW
            node(
                func=create_base_model, # This remains unchanged for SOC as it uses only the base model.
                inputs=["params:pretrained", "soc_weights"],
                outputs="soc_model",
                name="create_soc_model_node"
            ),
            node(
                func=prepare_meddra_dataset,
                inputs=["train_post_data_text_pre", "params:pretrained"],  # Assuming the config is saved in params
                outputs="meddra_dataset_train",
                name="prepare_meddra_dataset_node"
                ),
            node(
                func=create_dataloader,
                inputs=["meddra_dataset_train", "params:pretrained"],  # Assuming you have DataLoader config in params
                outputs="data_loader_train",
                name="create_dataloader_node"
            ),
            # node(
            #     func=create_extended_model,
            #     inputs=["params:pretrained", "pt_weights", "soc_weights"],
            #     outputs="pt_model",
            #     name="create_pt_model_node"
            # ),
            # node(
            #     func=create_extended_model,
            #     inputs=["params:pretrained", "llt_weights", "pt_weights", "soc_weights"],
            #     outputs="llt_model",
            #     name="create_llt_model_node"
            # ),
            node(
                func=train_single_model,
                inputs=["soc_model", "params:pretrained", "data_loader_train", "soc_weights"],
                outputs="trained_soc_model",
                name="train_soc_model_node"
            ),

            # # For PT model:
            # node(
            #     func=train_single_model,
            #     inputs=["pt_model", "params:epochs", "train_dataloader", "label_weights_pt", "trained_soc_model"],
            #     outputs="trained_pt_model",
            #     name="train_pt_model_node"
            # ),

            # # For LLT model:
            # node(
            #     func=train_single_model,
            #     inputs=["llt_model", "params:epochs", "train_dataloader", "label_weights_llt", "trained_pt_model"],
            #     outputs="trained_llt_model",
            #     name="train_llt_model_node"
            # )

        ]
    )
