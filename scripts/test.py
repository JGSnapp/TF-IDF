import pickle

gs = pickle.load(open("../models/GridSearch.pkl", "rb"))

while True:
    a = input()
    print(gs.predict([a]))
