# meddra_thesis


This project is master thesis software framework to classify hierchical classifcation. By utilizing kedros framework,  multiple pipelines ranging from data preprocessing to model training and evaluation are designed. This README provides an outline of the necessary steps to set up, run, test, and package your Kedro project.

# Setting Up the Project

Clone the project into local repo 

```

```
To install the necessary dependencies for this project, run:

```
pip install -r src/requirements.txt

```

# To execute all the pipelines 
```
kedro run
```
# To execute specific pipeline
```
kedro run --pipeline <pipeline_name>
```
# Project Pipelines Overview

In this project, multiple pipelines are orchestrated to perform distinct tasks, from data preprocessing to model training and evaluation. Here is a detailed breakdown of the available pipelines and their corresponding functionalities.

## Pipelines

### `data_abb` - Data Abbreviation Pipeline
**Description:**  
This pipeline focuses on shortening or simplifying textual data, making it more manageable and straightforward.

**Associated Pipeline Function:**  
- `dabb.create_data_abbreviation_pipeline()`

---

### `data_txt` - Text Preprocessing Pipeline
**Description:**  
Processes textual data to enhance its consistency and quality. The pipeline involves tasks such as label encoding, text lowercasing, and removal of special characters.

**Associated Pipeline Function:**  
- `dtxt.create_preprocessing_pipeline()`

---

### `data_aug` - Data Augmentation Pipeline
**Description:**  
Augments the dataset by applying various transformations, ensuring that the model is exposed to varied data patterns during training.

**Associated Pipeline Function:**  
- `daug.create_aug_pipeline()`

---

### `model_training` - Model Training Pipeline
**Description:**  
Focuses on training the models based on the cleaned and augmented data, ensuring optimal performance.

**Associated Pipeline Function:**  
- `model_training_pipeline()`

---

### `model_evaluation` - Model Evaluation Pipeline
**Description:**  
Evaluates the trained models using test datasets, generating performance metrics to gauge model effectiveness.

**Associated Pipeline Function:**  
- `model_evaluation_pipeline()`

---

### `model_pipeline` - Combined Model Training and Evaluation Pipeline
**Description:**  
A comprehensive pipeline combining both training and evaluation stages, allowing for an end-to-end execution of model processes.

**Associated Pipeline Function:**  
- Combination of `model_training_pipeline()` and `model_evaluation_pipeline()`

---

### `__default__` - Default Pipeline
**Description:**  
The default pipeline set for the project, which constitutes both the training and evaluation of models.

**Associated Pipeline Function:**  
- Summation of `model_training_pipeline()` and `model_evaluation_pipeline()`

---

## Usage

To invoke a particular pipeline, simply select its corresponding name when running the project. This allows for modularity and flexibility, as different stages of the project can be executed independently or in a sequence.

