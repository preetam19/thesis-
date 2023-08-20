# Pipeline Data Augmentation


## Overview

The `data_augmentation` pipeline is devised to augment the data by applying several techniques to increase the variety in the dataset and then split it into training and testing subsets. The augmentation techniques include:
- Shuffling the words in the text.
- Replacing words with their synonyms.
- Cropping the text to generate subsegments.

## Pipeline inputs

- `pre_processed_data`: Data that has already undergone the text preprocessing steps.
- `params:augmentation`: Configuration parameters to guide the augmentation process.

## Pipeline outputs

- `data_aug`: The augmented dataset.
- `train_post_data_text_pre`: Training subset after augmentation.
- `test_post_data_text_pre`: Testing subset after augmentation.

## Usage

Use the `create_aug_pipeline` function to generate the Kedro pipeline for data augmentation and splitting. This pipeline can be easily integrated into a broader Kedro project workflow.

## Important Notes

- The source dataset (`pre_processed_data`) should conform to the expected schema.
- Configuration parameters under `params:augmentation` must be provided to guide the augmentation process, such as deciding which augmentation techniques to use.
