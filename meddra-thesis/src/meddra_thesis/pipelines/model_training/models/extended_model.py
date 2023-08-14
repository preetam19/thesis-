import torch
import torch.nn as nn
from .base_model import BaseModel



class GeneralExtendedModel(BaseModel):
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
        
        # Pass through each hidden layer and layer norm based on k
        for i in range(self.k):
            output = self.hidden_layers[i](output)
            output = self.layer_norms[i](output)
            output = torch.relu(output)
        
        output_logits = self.output_layer(output)
        output_softmax = self.softmax(output_logits)
        return out, output_softmax, output_logits


# class GeneralExtendedModel(BaseModel):
#     def __init__(self, model, num_labels, *prev_label_nums):
#         super(GeneralExtendedModel, self).__init__(model, num_labels) 
#         self.model = model
#         total_input_size = self.model.config.hidden_size + sum(prev_label_nums)
#         self.hidden_layer = nn.Linear(total_input_size, self.model.config.hidden_size*2)
#         self.hidden_layer_2 = nn.Linear(self.model.config.hidden_size*2, self.model.config.hidden_size*4)
#         self.layer_norm_1 = nn.LayerNorm(self.model.config.hidden_size*2)
#         self.layer_norm_2 = nn.LayerNorm(self.model.config.hidden_size*4)
#         self.output_layer = nn.Linear(self.model.config.hidden_size*4, num_labels)

#     def forward(self, input_ids, attention_mask, token_type_ids, prev_logits):
#         out = self.model(input_ids, attention_mask, token_type_ids, return_dict =True)
#         output = self.drop(out['pooler_output'])
#         output = torch.cat([output, prev_logits], dim=-1)
#         output = self.hidden_layer(output)
#         output = self.layer_norm_1(output)
#         output = torch.relu(output)
        
#         output = self.hidden_layer_2(output)
#         output = self.layer_norm_2(output)
#         output = torch.relu(output)
        
#         output_logits = self.output_layer(output)
#         output_softmax = self.softmax(output_logits)
#         return out , output_softmax,  output_logits

# class GeneralExtendedModel(BaseModel):
#     def __init__(self, pretrained_model, num_labels, hidden_layers_sizes, *prev_label_nums):
#         super(GeneralExtendedModel, self).__init__(pretrained_model, num_labels)     
#         total_input_size = self.model.config.hidden_size + sum(prev_label_nums)

#         self.hidden_layers = nn.ModuleList([nn.Linear(total_input_size, hidden_layers_sizes[0])]) 
#         for i in range(0, len(hidden_layers_sizes)-1):
#             self.hidden_layers.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))      
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for size in hidden_layers_sizes])  
#         self.output_layer = nn.Linear(hidden_layers_sizes[-1], num_labels)

#     def forward(self, input_ids, attention_mask, token_type_ids, prev_logits=None):
#         _, _, output_logits = super().forward(input_ids, attention_mask, token_type_ids)
        
#         if prev_logits is not None:
#             print(prev_logits.shape)
#             print(output_logits.shape)
#             output = torch.cat([output_logits , prev_logits], dim=-1)
#         else:
#             output = output_logits
        
#         for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.layer_norms)):
#             output = layer(output)
#             output = norm(output)
#             output = torch.relu(output)
        
#         output_logits = self.output_layer(output)
#         output_softmax = self.softmax(output_logits)
#         return output_softmax, output_logits
