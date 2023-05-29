import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.classifiers.transformer_based.base import CustomBertClassifier
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.classifiers.transformer_based.constants import RANDOM_SEED
from src.classifiers.transformer_based.base import MeldDataset
from src.classifiers.utils import create_data_loader
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from matplotlib import pyplot as plt


class BertClassifierInterface:
    def __init__(self, model_name, dropout_prob, n_classes, lr=2e-5):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomBertClassifier(
            model_name=model_name,
            dropout_prob=dropout_prob,
            n_classes=n_classes
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.mapper = []
        self.history = []

    @staticmethod
    def data_preprocess(df: pd.DataFrame, tokenizer, x_label: str, y_label: str,
                 max_length: int, batch_size=16, transform=None):
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
        train_dataset = MeldDataset(df_train, tokenizer, x_label, y_label, max_length, transform)
        val_dataset = MeldDataset(df_val, tokenizer, x_label, y_label, max_length, transform)
        assert np.all(np.equal(train_dataset.categories[0], val_dataset.categories[0]))
        mapper = train_dataset.categories[0]
        train_data_loader: DataLoader = create_data_loader(train_dataset, batch_size)
        val_data_loader: DataLoader = create_data_loader(val_dataset, batch_size)
        return train_data_loader, val_data_loader, mapper

    def train(self, df: pd.DataFrame, x_label: str, y_label: str,
                 max_length: int = 70, batch_size: int = 16, epochs: int = 10, transform=None):
        train_data_loader, val_data_loader, self.mapper = self.data_preprocess(df, self.tokenizer, x_label, y_label, max_length, batch_size, transform)
        total_steps: int = len(train_data_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        best_acc: float = 0
        for epoch_i in range(epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            print('Training...')

            train_acc, train_loss = self._train(
                data_loader=train_data_loader,
                scheduler=scheduler
            )

            print("  Train accuracy: {0:.2f}".format(train_acc))
            print("  Train loss: {0:.2f}".format(train_loss))

            print('Running validation...')

            val_acc, val_loss = self.evaluate(
                data_loader=val_data_loader,
            )

            print("  Validation accuracy: {0:.2f}".format(val_acc))
            print("  Validation loss: {0:.2f}".format(val_loss))
            current_history = [train_acc, train_loss, val_acc, val_loss]

            self.history.append(current_history)

            # save model state with best accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'models/best_model.bin')

    def _train(self, data_loader: DataLoader, scheduler):

        model = self.model.train()

        losses = []
        correct_predictions: int = 0

        loop = tqdm(data_loader)
        for idx, d in enumerate(loop):
            input_ids = d[0].to(self.device)
            attention_mask = d[1].to(self.device)
            targets = d[2].to(self.device)

            # get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, predictions = torch.max(outputs, dim=1)
            _, correct = torch.max(targets, dim=1)
            correct_predictions += sum(torch.eq(predictions, correct))

            loss = self.loss_function(outputs, targets)
            losses.append(loss.item())

            # Backward prop
            loss.backward()

            # Gradient Descent
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()

        return float(correct_predictions) / len(data_loader.dataset), np.mean(losses)

    def evaluate(self, data_loader: DataLoader):
        # set mode
        model = self.model.eval()

        losses = []
        correct_predictions: int = 0

        with torch.no_grad():
            loop = tqdm(data_loader)
            for idx, d in enumerate(loop):
                input_ids = d[0].to(self.device)
                attention_mask = d[1].to(self.device)
                targets = d[2].to(self.device)

                # get model outputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs, dim=1)
                _, correct_preds = torch.max(targets, dim=1)
                correct_predictions += sum(torch.eq(preds, correct_preds))
                loss = self.loss_function(outputs, targets)
                losses.append(loss.item())

        return float(correct_predictions) / len(data_loader.dataset), np.mean(losses)

    def predict(self, data_loader: DataLoader):
        # set mode
        model = self.model.eval()

        x_values = []
        y_predictions = []
        y_probabilities = []
        y_actual = []

        with torch.no_grad():
            loop = tqdm(data_loader)
            for idx, d in enumerate(loop):
                input_ids = d[0].to(self.device)
                attention_mask = d[1].to(self.device)
                targets = d[2].to(self.device)
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

    def load(self, filename):
        self.model.load_state_dict(torch.load(f'models/{filename}.bin'))

    def save(self, filename="final_model"):
        torch.save(self.model.state_dict(), f'models/{filename}.bin')

    def visualize_training(self):
        hist_time = len(self.history)
        epochs = list(range(1, hist_time + 1))

        train_acc = [nested[0] for nested in self.history]
        train_loss = [nested[1] for nested in self.history]
        val_acc = [nested[2] for nested in self.history]
        val_loss = [nested[3] for nested in self.history]

        # Plotting these values
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.plot(epochs, val_loss, label='Validation Loss')

        # Adding a title
        plt.title('Bert Model Training')

        # Adding x and y label
        plt.xlabel('Epochs')
        plt.ylabel('Loss and Accuracy')

        # Add a legend
        plt.legend()

        # Displaying the plot
        plt.show()
