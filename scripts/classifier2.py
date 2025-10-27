import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

filename = '../models/GridSearch.pkl'

param_grid = {
'vectorizer__ngram_range': [(1, 1), (1, 2)],
'lr__C': [0.01, 0.1, 1, 10],
'lr__penalty': ['l2'],
'lr__solver': ['liblinear', 'lbfgs'],
}

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=5000))
])

train_df = pd.read_csv("../data/output/train.csv")
X_train = train_df['text']
Y_train = train_df['type']
test_df = pd.read_csv("../data/output/train.csv")
X_test = test_df['text']
Y_test = test_df['type']

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train, Y_train)
Y_pred = gs.predict(X_test)

acc  = accuracy_score(Y_test, Y_pred)
prec = precision_score(Y_test, Y_pred, average='weighted')
rec  = recall_score(Y_test, Y_pred, average='weighted')
f1   = f1_score(Y_test, Y_pred, average='weighted')

print(acc, prec, rec, f1)

pickle.dump(gs, open(filename, 'wb'))