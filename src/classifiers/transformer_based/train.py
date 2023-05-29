import pandas as pd
import numpy as np
from constants import *
from src.classifiers.utils import process_data, create_data_loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.classifiers.transformer_based.custom_model import BertClassifierInterface, MeldDataset


df: pd.DataFrame = pd.read_csv(DATA_PATH)
df = process_data(df, Y_LABEL, Y_CLASSES, TRANSCRIPT_MATCH_THRESHOLD, SAMPLE)

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

model = BertClassifierInterface(MODEL_NAME, DROPOUT_PROB, len(Y_CLASSES))
model.train(df_train, X_LABEL, Y_LABEL, MAX_LENGTH, BATCH_SIZE, EPOCHS)


print('Running test...')

test_dataset = MeldDataset(df_test, model.tokenizer, X_LABEL, Y_LABEL, MAX_LENGTH)
assert np.all(np.equal(test_dataset.categories[0], model.mapper))
mapper = test_dataset.categories
test_data_loader = create_data_loader(test_dataset, BATCH_SIZE)

test_acc, _ = model.evaluate(test_data_loader)

print("  Test accuracy: {0:.2f}".format(test_acc))

x_val, y_pred, y_probs, y_test = model.predict(test_data_loader)

print(classification_report(np.argmax(y_test, axis=1), y_pred))
