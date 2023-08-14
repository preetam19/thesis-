from transformers import AutoModel
from .models.extended_model import GeneralExtendedModel
from .models.base_model import BaseModel
import torch 
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import pickle
from torch.utils.data import DataLoader
import copy 
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_base_model(params, weights):
    model_name = params["model_name"]
    pretrained_model_instance = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return BaseModel(copy.deepcopy(pretrained_model_instance), len(weights))

def create_extended_model(params, weights, *prev_weights):
    model_name = params["model_name"]
    complexity = params["complexity"]
    pretrained_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    num_labels = len(weights)
    prev_label_nums = [len(weight) for weight in prev_weights]
    
    return GeneralExtendedModel(copy.deepcopy(pretrained_model), num_labels,complexity, *prev_label_nums)


def calculate_weights(df, config):
    weights = {}
    for col in config['cols']:
        label_list = df[col].values
        num_labels = df[col].nunique()
        class_frequencies = Counter(label_list)
        class_weights = [class_frequencies[i] for i in range(num_labels)]
        class_weights = np.array(class_weights) / len(label_list)
        weights[f"{col}_weights"] = class_weights
    return weights



def training_init(model, label_weight):
    print(f'training_init device: {device}')
    weights = torch.FloatTensor(label_weight).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=3e-5)
    return  loss_fn, optim

def train_single_model(model, config, data_loader_train, label_weight, *eval_models):
    eval_models_list = list(eval_models)
    loss_fn, optim = training_init(model, label_weight=label_weight)
    epochs = config['epochs']
    model.to(device)
    print(f'model_train script soc model is sent to {device}')
    model.train()
    eval_models = []
    correct_predictions = 0
    for epoch in tqdm(range(epochs)):
        for batch in data_loader_train:
            optim.zero_grad()
            input_ids = torch.squeeze(batch["input_ids"], (1)).to(device)
            attention_mask = torch.squeeze(batch["attention_mask"], (1)).to(device)
            token_type_ids = torch.squeeze(batch["token_type_ids"], (1)).to(device)
            labels = batch['soc_label'].to(device)
            
            pooled_outputs = []
            if eval_models_list:
                concatenated_pooled_outputs = None
                for idx, eval_model in enumerate(eval_models_list):
                    eval_model.eval().to(device)
                    if idx == 0:
                        _, _, pooled_output_eval = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    else:
                        _, _, pooled_output_eval = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)
                    
                    pooled_outputs.append(pooled_output_eval)
                    concatenated_pooled_outputs = torch.cat(pooled_outputs, dim=-1)
                
                _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)
            else:
                _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optim.step()
            
        print(f'LOSS: {loss} for the epoch: {epoch}')
    print(f'Predicted {correct_predictions} labels correctly')
    return model





















# def train_single_model(model, config, data_loader_train, label_weight, *eval_model):
#     loss_fn, optim = training_init(model, label_weight=label_weight)
#     epochs = config['epochs']
#     # if eval_model:
#     #     eval_model.eval()
#     model.to(device)
#     print(f'model_train script soc model is sent to {device}')
#     model.train()
#     correct_predictions = 0
#     for epoch in tqdm(range(epochs)):
#         for batch in data_loader_train:
#             optim.zero_grad()
#             input_ids = torch.squeeze(batch["input_ids"], (1)).to(device)
#             attention_mask = torch.squeeze(batch["attention_mask"], (1)).to(device)
#             token_type_ids = torch.squeeze(batch["token_type_ids"], (1)).to(device)
#             labels = batch['soc_label'].to(device)
            
#             pooled_outputs = []
#             if eval_model:
#                 for i in range(len(eval_model)):
#                     eval_model[i].eval().to(device)
#                     if i ==0:
#                         _, _, pooled_output_eval = eval_model[i](input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#                         pooled_outputs.append(pooled_output_eval)
#                     concatenated_pooled_outputs = torch.cat(pooled_outputs, dim=-1)   
#                     _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)
                
#                 concatenated_pooled_outputs = torch.cat(pooled_outputs, dim=-1)
#                 print(concatenated_pooled_outputs.shape)
#                 _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)
#             else:
#                 _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
#             _, preds = torch.max(outputs, dim=1)
#             correct_predictions += torch.sum(preds == labels)
#             loss = loss_fn(outputs, labels)

#             loss.backward()
#             optim.step()
            
#         print(f'LOSS: {loss} for the epoch: {epoch}')
#     print(f'Predicted {correct_predictions} labels correctly')
#     return model

























# def train_single_model(model, config, data_loader_train, label_weight, eval_model=None):
#     loss_fn, optim = training_init(model, label_weight=label_weight)
#     epochs = config['epochs']
#     if eval_model:
#         eval_model.eval()
#     model.to(device)
#     print(f'model_train scrip soc model is sent to {device}')
#     model.train()
#     correct_predictions = 0
#     for epoch in tqdm(range(epochs)):
#         for batch in data_loader_train:
#             optim.zero_grad()
#             input_ids = batch["input_ids"]
#             attention_mask = batch["attention_mask"]
#             token_type_ids = batch["token_type_ids"]
#             input_ids = torch.squeeze(input_ids, (1))
#             input_ids = torch.squeeze(input_ids, (1))
#             input_ids = torch.squeeze(input_ids, (1))
#             attention_mask = torch.squeeze(attention_mask, (1))
#             token_type_ids = torch.squeeze(token_type_ids, (1))
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             token_type_ids = token_type_ids.to(device)
#             labels = batch['soc_label'].to(device) 
#             logits = None
#             if eval_model:
#                 print(len(eval_model))
#                 model_output_eval, label_output_eval, pooled_output_eval= eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#                 _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,prev_logits= pooled_output_eval)
#             else:
#                 _,outputs,_ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

#             _, preds = torch.max(outputs, dim=1)
#             correct_predictions += torch.sum(preds == labels)
#             loss = loss_fn(outputs, labels)

#             loss.backward()
#             optim.step()

#         print(f'LOSS: {loss} for the epoch: {epoch}')
#     print(f'Predicted {correct_predictions} labels correctly')

#     return model


