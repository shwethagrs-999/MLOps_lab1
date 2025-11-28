import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

# FIX COLUMN SPACES
df.columns = df.columns.str.strip()

X = df[["feature1", "feature2"]]
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

print("Training complete")

