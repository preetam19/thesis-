
# Pipeline Model Training
## Overview
The model_training pipeline is dedicated to the construction, training, and evaluation of multi-task deep learning models tailored for MedDRA classification tasks. This pipeline encompasses the following main processes:

1. Weight calculation for different labels to handle class imbalance.
2. Initialization of both the base model, built upon pre-trained architectures, and the extended model which provides multi-task learning capabilities.
3. Preparation of the MedDRA dataset for model consumption.
4. Training of the models in a hierarchical manner, with outputs of preceding tasks being inputs for subsequent tasks.
5. Saving trained model states for future use or deployment.
6. The pipeline is designed to facilitate the hierarchical nature of MedDRA tasks and to harness the strengths of transfer learning with pre-trained models.

## Pipeline outputs
- `train_post_data_text_pre` : Preprocessed and cleaned data, ready for model training.
- `params` : A collection of configuration parameters, including training hyperparameters, model details, and dataset-specific configurations.

- Trained models for different tasks (trained_soc_model, trained_pt_model, trained_llt_model).
- A saved state for each trained model, making it convenient for deployment or further fine-tuning.
## Usage
To utilize this pipeline, employ the create_pipeline function. This constructs and provides the Kedro pipeline for model training. Integrate the resultant pipeline into the broader project workflow to ensure streamlined model training.

## Important Notes
- Before training, ensure the source data (train_post_data_text_pre) adheres to the anticipated schema, especially in terms of columns being processed.
- Training parameters and model details should be accurately configured in the params for optimal results.
- The pipeline is structured to leverage the hierarchical nature of MedDRA classification, so ensure the training data and the tasks are organized in this manner.
- The pipeline also incorporates checkpoints for model states, facilitating recovery and incremental training. Ensure the checkpoint paths and configurations are set as desired.