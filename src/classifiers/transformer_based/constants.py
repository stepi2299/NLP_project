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