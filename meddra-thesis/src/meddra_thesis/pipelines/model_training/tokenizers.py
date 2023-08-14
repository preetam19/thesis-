import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd


class MeddraDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.x = data[:, 1]
        self.labels_soc = data[:, 3]
        self.labels_pt = data[:, 2]
        self.labels_llt = data[:, 0]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = self.x[idx]
        label_soc = self.labels_soc[idx]
        label_pt = self.labels_pt[idx]
        label_llt = self.labels_llt[idx]
        text_tokenized = self.tokenizer(
            item,
            max_length=30,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_tokenized["input_ids"].flatten()
        attention_mask = text_tokenized["attention_mask"].flatten()
        token_type_ids = text_tokenized["token_type_ids"].flatten()

        return {
            "text": item,
            "llt_label": torch.tensor(label_llt, dtype=torch.long),
            "pt_label": torch.tensor(label_pt, dtype=torch.long),
            "soc_label": torch.tensor(label_soc, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


def prepare_meddra_dataset(df_train, config):
    df_train = pd.DataFrame(df_train)
    tokenizer = AutoTokenizer.from_pretrained(config['path'])
    df_train_array = df_train.to_numpy()
    return MeddraDataset(df_train_array, tokenizer)


def create_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config['batch_size'], num_workers=0, shuffle=True)