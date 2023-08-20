import torch
import torch.nn as nn
from .base_model import BaseModel



class GeneralExtendedModel(BaseModel):
    """
    An extended model class that builds upon the BaseModel. It incorporates additional functionality to 
    handle multi-task learning scenarios where the model is required to learn from both its current task 
    and previous tasks. The model dynamically adds hidden layers based on specified complexity, with each 
    layer followed by layer normalization. These hidden layers are designed to merge the features learned 
    from the pretrained model and the outputs (logits) from previous tasks. The final output is then produced 
    by an output layer tailored to the classification task.
    """
    def __init__(self, model, num_labels, complexity, *prev_label_nums):
        super(GeneralExtendedModel, self).__init__(model, num_labels)
        
        self.model = model
        self.k = complexity
        total_input_size = self.model.config.hidden_size + sum(prev_label_nums)
        
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        input_size = total_input_size
        for i in range(self.k):
            hidden_size = self.model.config.hidden_size * (2 ** (i + 1))
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            input_size = hidden_size
        
        self.output_layer = nn.Linear(input_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, prev_logits):
        out = self.model(input_ids, attention_mask, token_type_ids, return_dict=True)
        output = self.drop(out['pooler_output'])
        output = torch.cat([output, prev_logits], dim=-1)
        
        for i in range(self.k):
            output = self.hidden_layers[i](output)
            output = self.layer_norms[i](output)
            output = torch.relu(output)
        
        output_logits = self.output_layer(output)
        output_softmax = self.softmax(output_logits)
        return out, output_softmax, output_logits
