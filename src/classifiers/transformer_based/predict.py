import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import speech_recognition as sr

from base import CustomBertClassifier, predict, MeldDataset
from constants import *
from src.classifiers.utils import create_data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--state", "-s", type=str, required=True)
parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

print('Transcribing...')

transcription: str = ''
r = sr.Recognizer()
try:
    audio_file = os.path.join(args.file)
    audio = sr.AudioFile(audio_file)
    with audio as source:
        r.adjust_for_ambient_noise(audio)
        audio = r.record(source)
        transcription = r.recognize_whisper(audio)
except Exception as e:
    print(f"Something went wrong with audio, exception: {e}")
    exit(1)

print("  Transcribed text: {}".format(transcription))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = custom_model = CustomBertClassifier(
    model_name=MODEL_NAME,
    dropout_prob=DROPOUT_PROB,
    n_classes=len(Y_CLASSES)
)
model.load_state_dict(torch.load(args.state))

model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

d = {X_LABEL: [transcription], Y_LABEL: ['Unknown']}
df_test = pd.DataFrame(data=d)

test_dataset: Dataset = MeldDataset(df_test, tokenizer, X_LABEL, Y_LABEL,
                                    MAX_LENGTH)
test_data_loader: DataLoader = create_data_loader(test_dataset, BATCH_SIZE)

print('Predicting...')

_, y_pred, y_logits, _ = predict(
    model=custom_model,
    data_loader=test_data_loader,
    dev=device
)

print("  Class logits: {}".format(y_logits))
print("  Prediction: {}".format(y_pred))
