import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder


class MeldDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, x_label: str, y_label: str, max_length: int, augment=None):
        self.x_list: np.ndarray = df[x_label].to_numpy()
        ohe = OneHotEncoder()
        codes = df[y_label].to_numpy()
        codes = np.expand_dims(codes, axis=1)
        self.y_list: np.ndarray = ohe.fit_transform(codes).toarray()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        text = self.x_list[item]
        if self.augment:
            text = self.augment(text)
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
        return inputs_ids, attention_mask, y_tensor


def create_data_loader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


def prepare_dataset_based_on_class(df: pd.DataFrame, y_label: str, y_classes: list) -> pd.DataFrame:
    all_possibles_classes = df[y_label].unique()
    if len(all_possibles_classes) == len(y_classes):
        return df
    elif len(y_classes) < 2:
        print("Minimal number of analyzed class is 2, setting analyzed class into two -> positive and negative")
        y_classes = ['negative', 'positive']
    class_to_delete = set(all_possibles_classes) - set(y_classes)
    df = df.loc[df[y_label] != list(class_to_delete)[0]].reset_index(drop=True)
    return df
