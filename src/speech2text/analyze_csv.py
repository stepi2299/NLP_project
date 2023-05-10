import pandas as pd
import re
import numpy as np

algorithm = "google"

data = pd.read_csv(f"csv_files/{algorithm}.csv")
columns = data.columns
print(len(data))


def similarity_words(first_sentence, second_sentence):
    good_words = 0
    idx_lat_correct_word = 0
    for word in first_sentence:
        for i in range(len(second_sentence)-idx_lat_correct_word):
            if word == second_sentence[i]:
                good_words += 1
                idx_lat_correct_word = i
                break
    return round((good_words/len(first_sentence))*100, 2)

ident = 0
read_lines = 0
the_same_words = 0
percentage_correct_words = []

original = data[columns[0]].to_numpy()
transcript = data[columns[1]].to_numpy()
for origin, heard in zip(original, transcript):
    if isinstance(heard, str):
        read_lines += 1
        if origin.lower() == heard.lower():
            ident += 1
        original_words = re.findall(r'\w+', origin.lower())
        heard_words = re.findall(r'\w+', heard.lower())
        if original_words == heard_words:
            the_same_words += 1
        perc = similarity_words(original_words, heard_words)
        percentage_correct_words.append(perc)
    else:
        continue

print(f"Identical sentences: {ident}, ({round((ident/len(data))*100, 2)}%)")
print(f"read lines: {read_lines}, ({round((read_lines/len(data))*100, 2)}%)")
print(f"Number phrases with the same words: {the_same_words}, ({round((the_same_words/len(data))*100, 2)}%)")
print(f"mean percentage of correct words is {round(np.mean(percentage_correct_words), 2)}")
print(f"Mean time of transcripting {round(data[columns[4]].mean(), 2)}s, min time: {round(data[columns[4]].min(), 2)}s"
      f", max time: {round(data[columns[4]].max(), 2)}s")
