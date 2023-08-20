from kedro.pipeline import Pipeline, node
from functools import partial
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    encode_labels,
    lowercase_text,
    remove_special_characters_from_column
)


from functools import partial
from kedro.pipeline import Pipeline
from kedro.pipeline.node import node

def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    """
    Create a pipeline for the text preprocessing process.

    This pipeline chains multiple text preprocessing nodes:
    1. Encodes labels for the columns 'soc', 'pt', and 'llt'.
    2. Converts the 'llt_term' column text to lowercase.
    3. Removes special characters from the 'llt_term' column.

    Parameters:
    - **kwargs: Additional keyword arguments (not currently used in this function).

    Returns:
    - Pipeline: Kedro pipeline for text preprocessing.
    """

    return Pipeline([
        node(
            func=partial(encode_labels, columns=['soc', 'pt', 'llt']),  # Label encoding node
            inputs="abbreviated_data",
            outputs=f"text_pre_encoded_data",
            name=f"text_pre_encode_labels_node"
        ),
        node(
            func=partial(lowercase_text, column='llt_term'),  # Lowercase conversion node
            inputs=f"text_pre_encoded_data",
            outputs=f"text_pre_encoded_data_lower",
            name=f"text_pre_lowercase_text_node"
        ),
        node(
            func=partial(remove_special_characters_from_column, column='llt_term'),  # Special characters removal node
            inputs=f"text_pre_encoded_data_lower",
            outputs=f"pre_processed_data",
            name=f"text_pre_remove_special_characters_node"
        )
    ])













    # return Pipeline(
    #     [
    #         node(
    #             func=partial(encode_labels, columns = ['soc', 'pt', 'llt']),
    #             inputs='post_data',
    #             outputs="encoded_data",
    #             name="encode_labels_node",
    #         ),
    #         node(
    #             func =partial(lowercase_text, column = 'llt_term'),
    #             inputs = 'encoded_data',
    #             outputs="encoded_data_lower",
    #             name="encode_labels_node_text"
    #         ),  
    #         node(
    #             func =partial(remove_special_characters_from_column, column = 'llt_term'),
    #             inputs = 'encoded_data_lower',
    #             outputs="post_data_text_pre",
    #             name="encode_labels_node_text_sc"
    #         )
    #     ]
    # )