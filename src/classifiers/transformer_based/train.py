import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import CustomBertClassifier
from utils import create_dataset, create_data_loader


RANDOM_SEED = 42

# data
SAMPLE = 1000
X_LABEL = 'Utterance'
Y_LABEL = 'Sentiment'
Y_CLASSES = ['negative', 'positive']

# model
MODEL_NAME = 'bert-base-uncased'
DROPOUT_PROB = 0.3

# training
EPOCHS = 4
BATCH_SIZE = 16
MAX_LENGTH = 128


def train(model: nn.Module, data_loader: DataLoader, loss_fn, optim, dev: torch.device, sched, n_samples: int):
    # set mode
    model = model.train()

    losses = []
    correct_predictions = 0

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
        correct_predictions += sum(torch.eq(predictions, targets))

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


def evaluate(model: CustomBertClassifier, data_loader: DataLoader, loss_fn, dev: torch.device, n_samples: int):
    # set mode
    model = model.eval()

    losses = []
    correct_predictions = 0

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

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += sum(torch.eq(predictions, targets))

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

    return float(correct_predictions) / n_samples, np.mean(losses)


# data preparation
df: pd.DataFrame = pd.read_csv('data/meld.csv')
df = df.loc[df[Y_LABEL].isin(Y_CLASSES)].reset_index(drop=True)
df[Y_LABEL] = df[Y_LABEL].replace(Y_CLASSES[0], 0)
df[Y_LABEL] = df[Y_LABEL].replace(Y_CLASSES[1], 1)
df[Y_LABEL] = pd.to_numeric(df[Y_LABEL])

# limit dataframe length
if SAMPLE:
    df = df.head(SAMPLE)

df_train: pd.DataFrame
df_test: pd.DataFrame
df_val: pd.DataFrame

# split: 80%, 10%, 10%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_dataset: TensorDataset = create_dataset(df_train, tokenizer, X_LABEL, Y_LABEL, MAX_LENGTH)
val_dataset: TensorDataset = create_dataset(df_val, tokenizer, X_LABEL, Y_LABEL, MAX_LENGTH)
test_dataset: TensorDataset = create_dataset(df_test, tokenizer, X_LABEL, Y_LABEL, MAX_LENGTH)

train_data_loader: DataLoader = create_data_loader(train_dataset, BATCH_SIZE)
val_data_loader: DataLoader = create_data_loader(val_dataset, BATCH_SIZE)
test_data_loader: DataLoader = create_data_loader(test_dataset, BATCH_SIZE)

bert_model = BertModel.from_pretrained(MODEL_NAME)
custom_model = CustomBertClassifier(
    model_name=MODEL_NAME,
    dropout_prob=DROPOUT_PROB,
    n_classes=len(Y_CLASSES)
).to(device)

params: list[tuple] = list(custom_model.named_parameters())
optimizer = AdamW(custom_model.parameters(), lr=2e-5)

total_steps: int = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_function = nn.CrossEntropyLoss().to(device)
best_acc: float = 0

for epoch_i in range(EPOCHS):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))

    print('Training...')

    train_acc, train_loss = train(
        model=custom_model,
        data_loader=train_data_loader,
        loss_fn=loss_function,
        optim=optimizer,
        dev=device,
        sched=scheduler,
        n_samples=len(df_train)
    )

    print("  Train accuracy: {0:.2f}".format(train_acc))
    print("  Train loss: {0:.2f}".format(train_loss))

    print('Running validation...')

    val_acc, val_loss = evaluate(
        model=custom_model,
        data_loader=val_data_loader,
        loss_fn=loss_function,
        dev=device,
        n_samples=len(df_val)
    )

    print("  Validation accuracy: {0:.2f}".format(val_acc))
    print("  Validation loss: {0:.2f}".format(val_loss))

    # save model state with best accuracy
    if val_acc > best_acc:
        torch.save(custom_model.state_dict(), 'best_model.bin')
        best_acc = val_acc

# check model accuracy on test data
print('Running test...')

test_acc, _ = evaluate(
    model=custom_model,
    data_loader=test_data_loader,
    loss_fn=loss_function,
    dev=device,
    n_samples=len(df_test)
)

print("  Test accuracy: {0:.2f}".format(test_acc))
