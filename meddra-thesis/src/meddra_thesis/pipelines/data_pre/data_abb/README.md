# Pipeline Data Abbreviation


## Overview

The `data_abb` pipeline is responsible for processing raw data and transforming it using abbreviation methods. It is a part of the broader data preprocessing stage and specifically targets columns in datasets with potential abbreviation requirements.

This pipeline leverages a series of functions that:
- Split words based on a certain length criterion,
- Combine unique words from multiple columns,
- Filter out unwanted words and abbreviations,
- Produce a new dataset with abbreviated forms where necessary.

## Pipeline inputs

- `merged_raw_data`: This is the primary input dataset, typically a merged form of raw data that might contain long-form text or unabbreviated entries.
  
- `params:data_abbreviation`: Parameters used for the abbreviation process. It includes references to columns being processed and a list of unwanted words or abbreviations.

## Pipeline outputs

- `abbreviated_data`: The resulting dataset after applying the abbreviation logic. This dataset retains the structure of `merged_raw_data` but with certain columns transformed to have abbreviated content where applicable.
