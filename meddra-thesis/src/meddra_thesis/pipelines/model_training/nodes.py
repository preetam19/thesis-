from transformers import AutoModel
from .models.extended_model import GeneralExtendedModel
from .models.base_model import BaseModel
import torch 
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import pickle
from torch.utils.data import DataLoader

from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_base_model(params, weights):
    model_name = params["model_name"]
    options = params["options"]
    pretrained_model_instance = AutoModel.from_pretrained(model_name, **options)
    return BaseModel(pretrained_model_instance, len(weights))

def create_extended_model(params, weights, *prev_weights):
    model_name = params["model_name"]
    pretrained_model_instance = AutoModel.from_pretrained(model_name)

    hidden_sizes = params.get("hidden_sizes", [512])  # Default size is [512] if not provided
    num_labels = len(weights)

    prev_label_nums = [len(weight) for weight in prev_weights]
    
    return GeneralExtendedModel(pretrained_model_instance, num_labels, hidden_sizes, *prev_label_nums)


def calculate_weights(df, config):
    weights = {}
    for col in config['cols']:
        label_list = df[col].values
        num_labels = df[col].nunique()
        class_frequencies = Counter(label_list)
        class_weights = [class_frequencies[i] for i in range(num_labels)]
        class_weights = np.array(class_weights) / len(label_list)
        weights[f"{col}_weights"] = class_weights
        print(len(class_weights))
    return weights



def training_init(model, label_weight):
    print(f'training_init device: {device}')
    weights = torch.FloatTensor(label_weight).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    optim = torch.optim.RMSprop(model.parameters(), lr=3e-5)
    return  loss_fn, optim

def train_single_model(model, config, data_loader_train, label_weight, eval_model=None):
    loss_fn, optim = training_init(model, label_weight=label_weight)
    epochs = config['epochs']
    if eval_model:
        eval_model.eval()

    model.train()
    correct_predictions = 0
    for epoch in tqdm(range(epochs)):
        for batch in data_loader_train:
            optim.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch['soc_label'].to(device) 

            logits = None
            if eval_model:
                eval_model_output, eval_model_outputs, _ = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, logits=eval_model_outputs)
                logits = outputs
            else:
                _, outputs, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            loss = loss_fn(logits, labels)

            loss.backward()
            optim.step()

        print(f'LOSS: {loss} for the epoch: {epoch}')
    print(f'Predicted {correct_predictions} labels correctly')

    return model


