# Pipeline Text Preprocessing

## Overview

The `text_preprocessing` pipeline focuses on cleansing and standardizing textual data in preparation for subsequent analysis or modeling stages. This pipeline encompasses the following main tasks:
1. Label encoding of categorical columns such as 'soc', 'pt', and 'llt'.
2. Conversion of textual content in the 'llt_term' column to lowercase.
3. Removal of special characters from the 'llt_term' column.

The pipeline ensures that textual data is in a consistent and clean format, optimized for downstream tasks.

## Pipeline inputs

- `abbreviated_data`: This input dataset has undergone abbreviation transformations and is ready for further text preprocessing.

## Pipeline outputs

- `text_pre_encoded_data`: Dataset with label-encoded columns ('soc', 'pt', 'llt').
- `text_pre_encoded_data_lower`: Dataset with the 'llt_term' column transformed to lowercase.
- `pre_processed_data`: Final preprocessed dataset with special characters removed from the 'llt_term' column.

## Usage

To invoke this pipeline, use the `create_preprocessing_pipeline` function, which constructs and returns the Kedro pipeline for text preprocessing. The resulting pipeline can be integrated with the broader project's workflow.

## Important Notes

- Ensure that the source dataset (`abbreviated_data`) conforms to the expected schema, especially regarding the columns being processed.
- The order of operations is crucial; label encoding precedes the other two steps for efficiency and accuracy.
