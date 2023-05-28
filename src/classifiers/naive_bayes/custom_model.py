import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from src.classifiers.utils import create_bag_of_words
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class CustomNaiveBayes:
    def __init__(self):
        self.model: MultinomialNB = MultinomialNB()
        self.report: dict = {}

    @staticmethod
    def data_preprocess(df: pd.DataFrame, x_label: str, y_label: str):
        x, y = create_bag_of_words(df, x_label, y_label)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test

    def train(self, df: pd.DataFrame, x_label: str, y_label: str):
        x_train, x_test, y_train, y_test = self.data_preprocess(df, x_label, y_label)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        self.report = self.evaluate(y_test=y_test, y_pred=y_pred)

    def predict(self, x):
        return self.model.predict(x)

    def load(self, filename):
        self.model = pickle.load(open(f"models/{filename}.sav", 'rb'))

    def save(self, filename):
        pickle.dump(self.model, open(f"models/{filename}.sav", 'wb'))
        df = pd.DataFrame(self.report).transpose()
        df.to_csv(f"models/{filename}.csv")

    @staticmethod
    def evaluate(y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)
        conf_mat = confusion_matrix(y_test, y_pred)
        print(conf_mat)
        return classification_report(y_test, y_pred, output_dict=True)
