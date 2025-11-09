import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

filename_tfidf = "../models/logisticReg.pkl"
filename_lr = "../models/logisticReg.pkl"

train_df = pd.read_csv("../data/output/train.csv")
train = train_df["text"]
test_df = pd.read_csv("../data/output/train.csv")
test = test_df["text"]

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train)
Y_train = train_df["type"]
X_test = vectorizer.transform(test)
Y_test = test_df["type"]

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

acc = accuracy_score(Y_test, Y_pred)
prec = precision_score(Y_test, Y_pred, average="weighted")
rec = recall_score(Y_test, Y_pred, average="weighted")
f1 = f1_score(Y_test, Y_pred, average="weighted")

print(acc, prec, rec, f1)

pickle.dump(vectorizer, open(filename_tfidf, "wb"))
pickle.dump(lr, open(filename_lr, "wb"))
