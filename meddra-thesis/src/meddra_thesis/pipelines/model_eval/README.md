# Model Evaluation Pipeline

## Overview

The model evaluation pipeline focuses on assessing the performance of trained models on test datasets. It aims to gauge the effectiveness and accuracy of each model in terms of various metrics, providing insights into the model's strengths and areas of improvement. The pipeline encompasses tasks like loading pre-trained model weights, evaluating the models, and computing performance metrics.

## Functions

### `load_model_weights`

**Description:**  
Loads the pre-trained weights for the provided models from a saved state.

**Parameters:**  
- `config`: Dictionary containing the path to the saved model weights.
- `*models`: Models for which weights need to be loaded.

**Returns:**  
- List of models with their weights loaded.

---

### `compute_metrics`

**Description:**  
Computes multiple classification metrics like F1 score, precision, and recall for the provided true and predicted labels.

**Parameters:**  
- `true_labels`: Actual labels of the data.
- `predicted_labels`: Labels predicted by the model.

**Returns:**  
- Dictionary containing micro and macro averages of F1 score, precision, and recall.

---

### `eval_loop`

**Description:**  
Evaluates the models using the provided test dataset. Processes the dataset, makes predictions, and collects the results for each model.

**Parameters:**  
- `test_dataloader`: DataLoader containing the test dataset.
- `len_array`: Provides the length of the dataset used for calculating accuracy.
- `models`: List of models to be evaluated.

**Returns:**  
- Dictionary storing predicted labels, true labels, and correct predictions for each model. Additionally, prints performance metrics to the console.

## Usage

Integrate the model evaluation pipeline with the broader project workflow after the model training stage. Ensure you have the necessary dependencies installed and the saved model weights accessible for evaluation.

## Important Notes

- Ensure you have the saved model weights accessible at the path specified in the `config` for the `load_model_weights` function.
- Evaluation metrics can vary depending on the data distribution and model architecture. Always interpret the metrics in the context of the specific problem domain.
