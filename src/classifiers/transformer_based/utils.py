import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def create_dataset(df: pd.DataFrame, tokenizer, x_label: str, y_label: str, max_length: int):
    x_list: np.ndarray = df[x_label].to_numpy()
    y_list: np.ndarray = df[y_label].to_numpy()

    input_ids = []
    attention_masks = []

    for x in x_list:
        encoded_dict: dict = tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    y_tensor = torch.tensor(y_list)

    return TensorDataset(input_ids, attention_masks, y_tensor)


def create_data_loader(tds: TensorDataset, batch_size: int):
    return DataLoader(
        tds,
        sampler=RandomSampler(tds),
        batch_size=batch_size
    )