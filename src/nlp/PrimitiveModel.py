# Built-in imports
import json
import os
import pickle
import random
from typing import *

# Project imports


# External imports
import numpy as np
import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Path configurations
PATH_INTENT = "nlp/intents.json"
PATH_WORDS_DATA = "nlp/data/bag_of_words.pickle"
PATH_MODEL = "nlp/models/primitive.tflearn"

# Global configurations

# Stemmer
stemmer = LancasterStemmer()

# Global data variables
dictionary = []  # dictionary of words we've seen (unique)
intents = []  # list of intents
utterances = {}  # dict of {intent => [utterances...]}
responses = {}  # dict of {intent => [responses...]}
train_x, train_y = [], []  # training data lists

# Global model variables
model = None
model_changed = False


# TODO: allow user to train bot by typing in something then classifying it as X intent
#       NLP preprocessing - NLTK tokenize, stemming
#       organization intent-utterance

#######################
# DATA / FILE METHODS #
#######################

def load_or_generate_data(force_generate=False, save_data=True):
    """
    Load data if exists or generate data from intents

    Args:
        save_data (bool): whether to save the data to file
        force_generate (bool): set to True to force the re-generation of data
    """
    if not force_generate and load_data():
        return
    generate_data(save_data)


def generate_data(save_data=True):
    """
    Generate data from intents and load into global variables

    Args:
        save_data (bool): whether to save the data to file
    """
    global dictionary, intents, utterances, responses, train_x, train_y

    # Load intents file
    with open(PATH_INTENT) as f:
        data = json.load(f)

    dictionary = set()
    intents = []
    utterances = {}
    responses = {}
    train_x, train_y = [], []

    temp_x, temp_y = [], []
    for intent, intent_data in data.items():
        # Save intent
        intents.append(intent)
        # Get the list of patterns (utterances) in this intent
        patterns = intent_data["patterns"]

        # Process and save each pattern
        for sentence in patterns:
            # Save utterance
            utterances[intent] = utterances.get(intent, []) + [sentence]
            # Preprocess sentence
            words = preprocess(sentence)
            # Update dictionary
            dictionary.update(words)
            # Add training data
            temp_x.append(words)
            temp_y.append(intent)

        # Add responses to each intent
        responses[intent] = intent_data["responses"]

    # Now:
    # - temp_x contains a list of list of tokens
    # - temp_y contains a list of intents
    # - dictionary contains all tokens in all sentences
    # - responses contains intent => [responses...]

    # Convert dictionary to sorted list to keep ordering
    # TODO: in the future, if this project gets bigger, this will need optimization
    #       the bag-of-words array is really sparse, consists of mostly 0's and only few 1's
    #       compress by only remembering the location of 1's?
    dictionary = sorted(dictionary)

    # Create training data
    for i, tokens in enumerate(temp_x):
        # TODO: optimize by converting dictionary to dict with {token => index}? O(1) lookup cost
        #       convert back to list afterwards? not sure, just an idea
        #       cause right now the complexity is O(len(tokens) * len(dictionary)) => O(n^2)
        x = [1 if word in tokens else 0 for word in dictionary]
        y = [0] * len(data)  # how many classes (intents) there are
        y[intents.index(temp_y[i])] = 1
        train_x.append(x)
        train_y.append(y)

    # Training data is now in train_x and train_y

    # Save this bag-of-words training data for faster access in the future
    if save_data:
        with open(PATH_WORDS_DATA, "wb") as f:
            pickle.dump((dictionary, intents, utterances, responses, train_x, train_y), f)

    # We are done here


def load_data():
    """
    Load previously saved data into global variables

    Returns:
        bool: whether the load is successful
    """
    global dictionary, intents, utterances, responses, train_x, train_y

    # Check file existence and permissions
    if not os.path.isfile(PATH_WORDS_DATA) or not os.access(PATH_WORDS_DATA, os.R_OK):
        return False
    # Open file and load data
    with open(PATH_WORDS_DATA, "rb") as f:
        dictionary, intents, utterances, responses, train_x, train_y = pickle.load(f)
    return True


def add_utterance(intent, utterance):
    """
    Add a new utterance to the target intent

    Args:
        intent (str): intent to be modified
        utterance (str): utterance to be added
    """
    global model_changed
    assert intent in intents, f"Invalid intent \"{intent}\""
    with open(PATH_INTENT) as f:
        data = json.load(f)
    data[intent]["patterns"].append(utterance)
    with open(PATH_INTENT, "w") as f:
        json.dump(data, f, indent=4)

    model_changed = True


# TODO: RELOAD MODEL AND SET MODEL CHANGED TO FALSE

##########################
# NEURAL NETWORK METHODS #
##########################

def create_and_train_model(epochs=1000, save_model=True):
    """
    Create and train the neural network

    Args:
        epochs (int): number of epochs to train for
        save_model (bool): whether to save the trained model to file
    """
    global model, train_x, train_y
    model = None

    with tf.Graph().as_default():
        # Build model
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
        net = tflearn.regression(net)
        model = tflearn.DNN(net)

        # Train model
        model.fit(train_x, train_y, n_epoch=epochs, batch_size=8, show_metric=True)

        # Save model
        if save_model:
            model.save(PATH_MODEL)


def load_model():
    """ Load model from disk """
    # TODO: check correctness of this
    global model
    model = tflearn.DNN(None)
    model.load(PATH_MODEL)


def predict(message):
    """
    Generate a response from the input message

    Args:
        message (str): input message

    Returns:
        str: predicted response
    """
    assert model is not None, "Model must be initialized before predicting!"
    results = model.predict([bag_of_words(preprocess(message))])[0]
    index = np.argmax(results)
    intent = intents[index]

    if results[index] < 0.7:
        return "I didn't get that, try something else"
    return random.choice(responses[intent])


###################
# UTILITY METHODS #
###################

def preprocess(message):
    """
    Preprocesses the message into a list of tokens by tokenizing and stemming

    Args:
        message (str): message to preprocess

    Returns:
        List[str]: list of preprocessed tokens (tokenized and stemmed)
    """
    output = []
    # Tokenize message (split string into small tokens)
    words = nltk.word_tokenize(message)
    for word in words:
        # Stem each word (eg. flying becomes fly after stemming)
        word = stemmer.stem(word)
        # TODO: apply rules to filter tokens, simple rules for now
        if word in ",.?!~" or len(word) <= 1:
            continue
        output.append(word)
    return output


def bag_of_words(tokens):
    """
    Generates a bag-of-words representation of the token list

    Args:
        tokens (List[str]): list of preprocessed tokens (tokenized and stemmed)

    Returns:
        np.array: numpy array of the bag-of-words representation of the token list
    """
    global dictionary
    tokens = set(tokens)
    # TODO: again, this could be faster with an improved dictionary data structure
    #       Currently O(len(dictionary)), can be improved to O(len(tokens))
    bag = [1 if word in tokens else 0 for word in dictionary]
    assert len(bag) == len(dictionary), "INVALID BAG-OF-WORDS MODEL! SOMETHING REALLY BAD HAPPENED!"
    return np.array(bag)


if __name__ == "__main__":
    # Change path
    PATH_INTENT = "intents.json"
    PATH_WORDS_DATA = "data/bag_of_words.pickle"
    PATH_MODEL = "models/primitive.tflearn"

    # Generate data
    print(f"Generating training data... ", end="")
    generate_data()
    print(f"OK!")

    # Train new model
    print(f"Creating and training the model... ", end="")
    create_and_train_model()
    print(f"OK!")

    # Test the model
    while True:
        message = input("You: ").lower()
        if message == "quit":
            break

        response = predict(message)
        print(response)
