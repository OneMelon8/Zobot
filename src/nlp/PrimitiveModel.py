# Built-in imports
import json
import pickle
import random

# Project imports


# External imports
import numpy as np
import tflearn
import tensorflow as tf

import nltk
from nltk.stem.lancaster import LancasterStemmer

# Stemmer
stemmer = LancasterStemmer()

# Intent dict:
# INTENT_NAME => { PATTERNS, RESPONSES }
intents = {}

# Dictionary of words we've seen
dictionary = []

train_x, train_y = [], []


# TODO: allow user to train bot by typing in something then classifying it as X intent
# NLP preprocessing - NLTK tokenize, stemming
# organization intent-utterance


def process_raw_data():
    global dictionary, intents, train_x, train_y

    # Load intents file
    with open("intents.json") as f:
        data = json.load(f)

    dictionary = set()
    temp_x = []
    temp_y = []
    for intent, intent_data in data.items():
        patterns = intent_data["patterns"]  # such as "what's up"
        words = set()
        for sentence in patterns:
            words.update(tokenize(sentence))
        # Update total dictionary
        dictionary.update(words)
        # Add training data
        temp_x.append(words)
        temp_y.append(intent)


def load_data():
    global dictionary, intents, train_x, train_y
    with open("nlp/data/words.pickle", "rb") as f:
        dictionary, intents, train_x, train_y = pickle.load(f)


def tokenize(message):
    words = nltk.word_tokenize(message)
    output = []
    for word in words:
        word = stemmer.stem(word)
        # TODO: apply rules to filter tokens, simple rules for now
        if word in ",.?!()\"" or len(word) <= 1:
            continue
        output.append(word)
    return output


def bag_of_words(tokens):
    global dictionary
    # TODO: bag of words

    return np.array([])


process_raw_data()
