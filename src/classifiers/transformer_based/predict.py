import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import speech_recognition as sr

from src.classifiers.transformer_based.constants import *
from src.classifiers.transformer_based.custom_model import BertClassifierInterface
from src.classifiers.transformer_based.dataset import MeldDataset
from src.classifiers.utils import create_data_loader

CLASS_MAPPER = {0: "negative", 1: "neutral", 2: "positive"}

parser = argparse.ArgumentParser()
parser.add_argument("--model_state", "-ms", type=str, required=True)
parser.add_argument("--input", "-i", type=str, required=True)
parser.add_argument("--model_type", "-mt", type=str, choices=["bert", "tiny_bert", "distil_bert"], default='bert')
args = parser.parse_args()

transcription: str = ''
r = sr.Recognizer()
try:
    audio_file = os.path.join(args.input)
    audio = sr.AudioFile(audio_file)
    with audio as source:
        r.adjust_for_ambient_noise(audio)
        audio = r.record(source)
        transcription = r.recognize_whisper(audio)
except Exception as e:
    print(f"Something went wrong with audio, exception: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_interface = BertClassifierInterface(args.model_type, DROPOUT_PROB, len(Y_CLASSES))

model_interface.model.load_state_dict(torch.load(args.model_state, map_location=device))

model_interface.model = model_interface.model.to(device)
model_interface.model.eval()

d = {X_LABEL: [transcription], Y_LABEL: ['unknown']}
df_test = pd.DataFrame(data=d)

test_dataset: MeldDataset = MeldDataset(df_test, model_interface.tokenizer, X_LABEL, Y_LABEL,
                                        MAX_LENGTH)
test_data_loader: DataLoader = create_data_loader(test_dataset, BATCH_SIZE)

print('Predicting...')

_, y_pred, y_logits, _ = model_interface.predict(data_loader=test_data_loader)

print("Transcription: {}".format(transcription))
print("Sentiment: {}".format(CLASS_MAPPER.get(y_pred.numpy()[0])))
print("Logits: {}".format(y_logits))
