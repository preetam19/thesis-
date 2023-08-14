import torch
import torch.nn as nn
from .base_model import BaseModel
class GeneralExtendedModel(BaseModel):
    def __init__(self, pretrained_model, num_labels, hidden_layers_sizes, *prev_label_nums):
        super(GeneralExtendedModel, self).__init__(pretrained_model, num_labels)     
        total_input_size = self.model.config.hidden_size + sum(prev_label_nums)

        self.hidden_layers = nn.ModuleList([nn.Linear(total_input_size, hidden_layers_sizes[0])]) 
        for i in range(0, len(hidden_layers_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))      
        self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for size in hidden_layers_sizes])  
        self.output_layer = nn.Linear(hidden_layers_sizes[-1], num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, *prev_logits):
        _, _, output_logits = super().forward(input_ids, attention_mask, token_type_ids)
        
        if prev_logits:
            output = torch.cat([output_logits] + list(prev_logits), dim=-1)
        else:
            output = output_logits
        
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.layer_norms)):
            output = layer(output)
            output = norm(output)
            output = torch.relu(output)
        
        output_logits = self.output_layer(output)
        output_softmax = self.softmax(output_logits)
        return output_softmax, output_logits
