import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, \
    get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from base import CustomBertClassifier, MeldDataset, train, evaluate, predict
from utils import create_data_loader, process_data

RANDOM_SEED = 42

DATA_PATH = "../../../data/meld.csv"

# data
SAMPLE = 100
X_LABEL = 'Transcription'
Y_LABEL = 'Sentiment'
Y_CLASSES = ['negative', 'positive', "neutral"]
TRANSCRIPT_MATCH_THRESHOLD = 0.2

# model
MODEL_NAME = 'bert-base-uncased'
DROPOUT_PROB = 0.3

# training
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 70

# data preparation
df: pd.DataFrame = pd.read_csv(DATA_PATH)
df = process_data(df, Y_LABEL, Y_CLASSES, SAMPLE, TRANSCRIPT_MATCH_THRESHOLD)

# split: 80%, 10%, 10%
df_train, df_test = train_test_split(df, test_size=0.2,
                                     random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5,
                                   random_state=RANDOM_SEED)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_dataset: Dataset = MeldDataset(df_train, tokenizer, X_LABEL, Y_LABEL,
                                     MAX_LENGTH)
val_dataset: Dataset = MeldDataset(df_val, tokenizer, X_LABEL, Y_LABEL,
                                   MAX_LENGTH)
test_dataset: Dataset = MeldDataset(df_test, tokenizer, X_LABEL, Y_LABEL,
                                    MAX_LENGTH)

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
        best_acc = val_acc
        torch.save(custom_model.state_dict(), 'best_model.bin')

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

x_val, y_pred, y_probs, y_test = predict(
    model=custom_model,
    data_loader=test_data_loader,
    dev=device
)

print(classification_report(np.argmax(y_test, axis=1), y_pred))
