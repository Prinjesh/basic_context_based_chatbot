import random
import json
from gtts import gTTS
import torch
import os
from model import NeuralNet
import speech_recognition as sr
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# preparation for the input data
intents_path = '/home/prinjesh/Downloads/Internship/chatbot/chatbot/intents.json'

with open(intents_path, 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
vector = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


# preprocessing functions for natural language inputs
def tokenize(input):
    tokens = nltk.word_tokenize(input)
    return tokens


def stem(input):
    stemmed = stemmer.stem(input.lower())
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


def chatbot():
    print("Let's chat! (Speak 'bye' to exit)")
    while True:
        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        # recognize speech using Google Speech Recognition
        try:
            print("You: " + r.recognize_google(audio))
        except sr.UnknownValueError:
            print("I could not recognise what you said. Can you speak a bit slow?")
            break
        except sr.RequestError as e:
            print(
                "Internet connection is not available right now to recognise your speech; {0}".format(e))
            break

        sentence = r.recognize_google(audio)
        print('Loop cleared!')
        if sentence == "bye":
            txt = "Good bye, Nice to meet you!"
            speach = gTTS(txt, lang='en', slow=False)
            speach.save('audio.mp3')
            os.system(
                '/home/prinjesh/Downloads/Internship/chatbot/chatbot/audio.mp3')
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, vector)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print(prob)
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"Maric: {random.choice(intent['responses'])}")
                    txt = random.choice(intent['responses'])
                    speach = gTTS(txt, lang='en', slow=False)
                    speach.save('audio.mp3')
                    os.system(
                        '/home/prinjesh/Downloads/Internship/chatbot/chatbot/audio.mp3')
        else:
            print("Maric: Your request is out of my scope")


chatbot()
