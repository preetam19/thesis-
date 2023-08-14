from kedro.pipeline import Pipeline, node
from functools import partial
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    encode_labels,
    lowercase_text,
    remove_special_characters_from_column
)

def create_generic_pipeline(input_data_name, prefix):
    return Pipeline([
        node(
            func=partial(encode_labels, columns=['soc', 'pt', 'llt']),
            inputs=input_data_name,
            outputs=f"{prefix}_encoded_data",
            name=f"{prefix}_encode_labels_node"
        ),
        node(
            func=partial(lowercase_text, column='llt_term'),
            inputs=f"{prefix}_encoded_data",
            outputs=f"{prefix}_encoded_data_lower",
            name=f"{prefix}_lowercase_text_node"
        ),
        node(
            func=partial(remove_special_characters_from_column, column='llt_term'),
            inputs=f"{prefix}_encoded_data_lower",
            outputs=f"{prefix}_post_data_text_pre",
            name=f"{prefix}_remove_special_characters_node"
        )
    ])

def create_preprocessing_pipeline(**kwargs) -> Pipeline:

    pipeline = create_generic_pipeline('post_data_train', 'preprocess')

    return pipeline 














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