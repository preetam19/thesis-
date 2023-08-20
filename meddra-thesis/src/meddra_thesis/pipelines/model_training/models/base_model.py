import torch 
import torch.nn as nn

class BaseModel(nn.Module):
    """ 
    A base model class that wraps around a pretrained model (e.g., BERT, BIOBERT, BlueBERT) and extends its capabilities 
    to perform classification tasks. The model's output is passed through a dropout layer for regularization, 
    a layer normalization for stabilizing the activations, and finally a linear layer to produce logits for 
    classification. The softmax function then computes the probability distribution of the classes.
    """
    def __init__(self, pretrained_model, num_labels):
        super(BaseModel, self).__init__()
        self.model = pretrained_model
        self.num_labels = num_labels
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.model.config.hidden_size, num_labels)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        model = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        output = self.drop(model["pooler_output"])
        output = self.layer_norm(output)
        output_logits = self.out(output)
        output = self.softmax(output_logits)
        return model , output, output_logits
 