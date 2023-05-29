import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class CustomBertClassifier(nn.Module):
    # bert core + dropout + one layer feed-forward
    def __init__(self, model_name, dropout_prob, n_classes):
        super(CustomBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.classifier(output)


class MeldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, x_label: str, y_label: str,
                 max_length: int, transform=None):
        self.x_list: np.ndarray = df[x_label].to_numpy()
        ohe = OneHotEncoder()
        codes = df[y_label].to_numpy()
        codes = np.expand_dims(codes, axis=1)
        self.y_list: np.ndarray = ohe.fit_transform(codes).toarray()
        self.categories = ohe.categories_
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        text = self.x_list[item]
        if self.transform:
            text = self.transform(text)
        encoded_dict: dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        inputs_ids = encoded_dict['input_ids'].reshape(-1)
        attention_mask = encoded_dict['attention_mask'].reshape(-1)
        y_tensor = torch.tensor(self.y_list[item])
        return inputs_ids, attention_mask, y_tensor, text

