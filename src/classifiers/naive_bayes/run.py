import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import process_data

DATA_PATH = "../../../data/meld.csv"
X_LABEL = 'Transcription'
Y_LABEL = 'Sentiment'
Y_CLASSES = ['negative', 'positive', "neutral"]
TRANSCRIPT_MATCH_THRESHOLD = 0.2

df: pd.DataFrame = pd.read_csv(DATA_PATH)
df = process_data(df, Y_LABEL, Y_CLASSES, TRANSCRIPT_MATCH_THRESHOLD)

count_vect: CountVectorizer = CountVectorizer()
bag_of_words = count_vect.fit_transform(df[X_LABEL])
bag_of_words: pd.DataFrame = pd.DataFrame(bag_of_words.toarray(),
                                          columns=count_vect.get_feature_names_out())

X: pd.DataFrame = bag_of_words
Y: pd.DataFrame = df[Y_LABEL]

x_train: pd.DataFrame
x_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

model: MultinomialNB = MultinomialNB()
model.fit(x_train, y_train)

y_pred: np.ndarray = model.predict(x_test)

print(classification_report(y_test, y_pred))
