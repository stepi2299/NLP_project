import pandas as pd
from src.classifiers.utils import process_data
from src.classifiers.naive_bayes.custom_model import CustomNaiveBayes


DATA_PATH = "../../../data/meld.csv"
X_LABEL = 'Transcription'
Y_LABEL = 'Sentiment'
Y_CLASSES = ['negative', 'positive', "neutral"]
FILENAME = 'test_model'
TRANSCRIPT_MATCH_THRESHOLD = 0.2

df: pd.DataFrame = pd.read_csv(DATA_PATH)
df = process_data(df, Y_LABEL, Y_CLASSES, TRANSCRIPT_MATCH_THRESHOLD)
model = CustomNaiveBayes()
model.train(df=df, x_label=X_LABEL, y_label=Y_LABEL)
model.save(FILENAME)

# model.load(FILENAME)
# _, x_test, _, y_test = model.data_preprocess(df, X_LABEL, Y_LABEL)
# y_pred = model.predict(x_test)
# model.evaluate(y_test=y_test, y_pred=y_pred)
