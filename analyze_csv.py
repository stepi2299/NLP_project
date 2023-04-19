import pandas as pd

data = pd.read_csv("data/heard_text.csv")
col1 = data["read"].to_numpy()
good = 0
for col in col1:
    if col == 1:
        good += 1
print("Read files: ", good, "percentage: ", good/len(col1), "all", len(col1))

col2 = data["identical"].to_numpy()
ident = 0
for col in col2:
    if col == 1:
        ident += 1

print("Identical transcipts: ", ident)
