from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import torch

from huggingface_hub import hf_hub_download
import fasttext
from navec import Navec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("../data/output/train.csv")
test_data = pd.read_csv("../data/output/test.csv")

train_texs = train_data["text"].to_list()
test_texs = test_data["text"].to_list()

y_train = train_data["type"].to_list()
y_test = test_data["type"].to_list()

train_texts_tokenized = [text.lower().split() for text in train_texs]
test_texts_tokenized = [text.lower().split() for text in test_texs]

word2vec = Word2Vec(train_texts_tokenized, vector_size=300, window=5, min_count=1)


def vectorize_sentence(tokens):
    vecs = [word2vec.wv[w] for w in tokens if w in word2vec.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)


X_train = np.array([vectorize_sentence(text) for text in train_texts_tokenized])
X_test = np.array([vectorize_sentence(text) for text in test_texts_tokenized])

logreg = LogisticRegression(max_iter=3000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(accuracy, f1)

path = "../navec_hudlit_v1_12B_500K_300d_100q.tar"
navec = Navec.load(path)


def vectorize_sentence_2(sentence):
    vecs = [navec.get(word) for word in sentence if word in navec]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)


X_train_navec = np.array([vectorize_sentence_2(text) for text in train_texts_tokenized])
X_test_navec = np.array([vectorize_sentence_2(text) for text in test_texts_tokenized])

logreg = LogisticRegression(max_iter=3000)
logreg.fit(X_train_navec, y_train)
y_pred_navec = logreg.predict(X_test_navec)

accuracy = accuracy_score(y_test, y_pred_navec)
f1 = f1_score(y_test, y_pred_navec, average="macro")
print(accuracy, f1)

fasttext_model = FastText(
    sentences=train_texts_tokenized, vector_size=300, window=5, min_count=1
)


def vectorize_sentence_3(tokens):
    vecs = [fasttext_model.wv[w] for w in tokens if w in fasttext_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)


X_train_fasttext = np.array(
    [vectorize_sentence_3(text) for text in train_texts_tokenized]
)
X_test_fasttext = np.array(
    [vectorize_sentence_3(text) for text in test_texts_tokenized]
)

logreg = LogisticRegression(max_iter=3000)
logreg.fit(X_train_fasttext, y_train)
y_pred_fasttext = logreg.predict(X_test_fasttext)

accuracy = accuracy_score(y_test, y_pred_fasttext)
f1 = f1_score(y_test, y_pred_fasttext, average="macro")
print(accuracy, f1)

path = hf_hub_download("facebook/fasttext-ru-vectors", filename="model.bin")
ft = fasttext.load_model(path)


def vectorize_sentence_4(sentence):
    vecs = [ft.get_word_vector(word) for word in sentence if word in ft.words]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)


X_train_fasttext2 = np.array(
    [vectorize_sentence_4(text) for text in train_texts_tokenized]
)
X_test_fasttext2 = np.array(
    [vectorize_sentence_4(text) for text in test_texts_tokenized]
)

logreg = LogisticRegression(max_iter=3000)
logreg.fit(X_train_fasttext2, y_train)
y_pred_fasttext2 = logreg.predict(X_test_fasttext2)

accuracy = accuracy_score(y_test, y_pred_fasttext2)
f1 = f1_score(y_test, y_pred_fasttext2, average="macro")
print(accuracy, f1)
