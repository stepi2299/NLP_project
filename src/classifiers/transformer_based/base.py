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
    def __init__(self, model_name, dropout_prob, n_classes=2):
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
                 max_length: int, augment=None):
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
        return inputs_ids, attention_mask, y_tensor, text


def train(model: nn.Module, data_loader: DataLoader, loss_fn, optim,
          dev: torch.device, sched, n_samples: int):
    # set mode
    model = model.train()

    losses = []
    correct_predictions: int = 0

    loop = tqdm(data_loader)
    for idx, d in enumerate(loop):
        input_ids = d[0].to(dev)
        attention_mask = d[1].to(dev)
        targets = d[2].to(dev)

        # get model outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, predictions = torch.max(outputs, dim=1)
        _, correct = torch.max(targets, dim=1)
        correct_predictions += sum(torch.eq(predictions, correct))

        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        # Backward prop
        loss.backward()

        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

    return float(correct_predictions) / n_samples, np.mean(losses)


def evaluate(model: CustomBertClassifier, data_loader: DataLoader, loss_fn,
             dev: torch.device, n_samples: int):
    # set mode
    model = model.eval()

    losses = []
    correct_predictions: int = 0

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, d in enumerate(loop):
            input_ids = d[0].to(dev)
            attention_mask = d[1].to(dev)
            targets = d[2].to(dev)

            # get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            _, correct_preds = torch.max(targets, dim=1)
            correct_predictions += sum(torch.eq(preds, correct_preds))
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

    return float(correct_predictions) / n_samples, np.mean(losses)


def predict(model: CustomBertClassifier, data_loader: DataLoader,
            dev: torch.device):
    # set mode
    model = model.eval()

    x_values = []
    y_predictions = []
    y_probabilities = []
    y_actual = []

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, d in enumerate(loop):
            input_ids = d[0].to(dev)
            attention_mask = d[1].to(dev)
            targets = d[2].to(dev)
            x_vals = d[3]

            # get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            x_values.extend(x_vals)
            y_predictions.extend(preds)
            y_probabilities.extend(outputs)
            y_actual.extend(targets)

    y_predictions = torch.stack(y_predictions).cpu()
    y_probabilities = torch.stack(y_probabilities).cpu()
    y_actual = torch.stack(y_actual).cpu()

    return x_values, y_predictions, y_probabilities, y_actual
