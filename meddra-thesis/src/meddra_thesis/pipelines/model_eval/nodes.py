"""
This is a boilerplate pipeline 'model_eval'
generated using Kedro 0.18.3
"""
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torch


def load_model_weights(config, *models):
    state_dicts = torch.load(config['save_model_path'])
    
    loaded_models = list(models)
    for model in loaded_models:
        model_key = f'model_{model.name}_state_dict'
        if model_key in state_dicts:
            model.load_state_dict(state_dicts[model_key])
    
    return loaded_models



def forward_pass(model, input_data):
    input_ids, attention_mask, token_type_ids = input_data
    outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    _, preds = torch.max(outputs, dim=1)
    return preds

def compute_metrics(true_labels, predicted_labels):
    metrics = {
        "f1_micro": f1_score(true_labels, predicted_labels, average='micro'),
        "precision_micro": precision_score(true_labels, predicted_labels, average='micro'),
        "recall_micro": recall_score(true_labels, predicted_labels, average='micro'),
        "f1_macro": f1_score(true_labels, predicted_labels, average='macro'),
        "precision_macro": precision_score(true_labels, predicted_labels, average='macro'),
        "recall_macro": recall_score(true_labels, predicted_labels, average='macro')
    }
    return metrics

def eval_model_with_dependencies(model, input_data, *pooled_outputs_from_prev_models):
    input_ids, attention_mask, token_type_ids = input_data
    concatenated_pooled_outputs = None

    if pooled_outputs_from_prev_models:
        concatenated_pooled_outputs = torch.cat(pooled_outputs_from_prev_models, dim=-1)
        _, outputs, pooled_output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)
    else:
        _, outputs, pooled_output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    _, preds = torch.max(outputs, dim=1)
    return preds, pooled_output

def eval_loop(test_dataloader, len_array, models):
    device = torch.device("cuda")

    print(f'in model eval device is {device}')
    
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = torch.squeeze(batch["input_ids"], 1).to(device)
            attention_mask = torch.squeeze(batch["attention_mask"], 1).to(device)
            token_type_ids = torch.squeeze(batch["token_type_ids"], 1).to(device)

            pooled_outputs = []
            for idx, model in enumerate(models):
                model.to(device)
                model.eval()

                if idx == 0:
                    _, outputs, new_pooled_output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    concatenated_pooled_outputs = torch.cat(pooled_outputs, dim=-1)
                    _, outputs, new_pooled_output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, prev_logits=concatenated_pooled_outputs)

                if new_pooled_output is not None:
                    pooled_outputs.append(new_pooled_output)

                model_name = model.name
                true_labels = batch[f'{model_name}_label'].to(device)
                preds = torch.argmax(outputs, dim=1)
                
                results[model_name]['correct_predictions'].append(torch.sum(preds == true_labels).item())
                results[model_name]['true_labels'].extend(true_labels.cpu().numpy())
                results[model_name]['pred_labels'].extend(preds.cpu().numpy())

    # Calculate metrics for each model
    for model in models:
        model_name = model.name
        accuracy = sum(results[model_name]['correct_predictions']) / len_array
        metrics = compute_metrics(results[model_name]['true_labels'], results[model_name]['pred_labels'])

        print(f'{model_name} ACCURACY: {accuracy}')
        for metric_name, metric_value in metrics.items():
            print(f'{model_name} {metric_name.upper()}: {metric_value}')
    
    return results
