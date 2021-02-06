# Built-in imports
import json
import pickle
import random

# External imports
import numpy as np
import tflearn
import tensorflow as tf

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
with open("nlp/intents.json") as f:
    data = json.load(f)

try:
    with open("nlp/data/words.pickle", "rb") as f:
        dictionary, labels, training, output = pickle.load(f)
except:
    print(f"RELOADING DATA!")
    dictionary = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            dictionary += wrds
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    dictionary = [stemmer.stem(word.lower()) for word in dictionary if word not in "?!.,"]
    dictionary = sorted(set(dictionary))
    labels.sort()

    # Bag of words
    training = []
    output = []

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in dictionary:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = [0] * len(labels)
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    with open("nlp/data/words.pickle", "wb") as f:
        pickle.dump((dictionary, labels, training, output), f)

# Training data
training = np.array(training)
output = np.array(output)

# TF learn
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("nlp/models/temp")
except:
    print(f"TRAINING NEW MODEL!")
    # Train if needed
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # Save
    model.save("nlp/models/temp")


def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Say something")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, dictionary)])
        index = np.argmax(results)
        tag = labels[index]

        if results[index] < 0.8:
            print("I didn't get that, try something else")
            continue
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        print(random.choice(responses))


chat()
# EOF
