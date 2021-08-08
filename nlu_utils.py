"""
Reference:
The code has been created refering to
below listed Github repository.
https://github.com/jerrytigerxu/Simple-Python-Chatbot
"""

import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens


def stem(word):
    stemmed = stemmer.stem(word.lower())
    return stemmed


def bag_of_words(tokens, words):
    # stem each word
    sentence_words = [stem(word) for word in tokens]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
