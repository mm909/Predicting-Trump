from flask import Flask
from keras.models import load_model
import tensorflow as tf
import numpy as np
import heapq
from flask import render_template
from flask import jsonify
from flask import request

app = Flask(__name__)

def load_keras_model():
    # return load_model('D:/Predictive-Text/best/Trump.hdf5')

    return load_model('D:/Predictive-Text/experiments/predictiveTrump/20200327-095125/weights/weights-improvement-04-0.6693.hdf5')
    return load_model('D:/Predictive-Text/experiments/predictiveTrump/20200327-131735/weights/weights-improvement-04-0.6673.hdf5')
    return load_model('D:/Predictive-Text/experiments/predictiveTrump/20200329-145328/weights/weights-improvement-04-0.6736.hdf5')

model = load_keras_model()

SEQUENCE_LENGTH = 40

# Gettings data
speeches = '../data/trump/speeches/clean/cleanSpeech.txt'
tweets = '../data/trump/tweets/clean/cleanTweets.txt'
text = open(tweets, encoding="utf8").read().lower()
text = text + open(speeches, encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def temperatureSample(preds, temperature=0.01):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    return x

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


def genSentence(text, words = 2):
    textOG = text
    text = text.lower()
    while len(text) < SEQUENCE_LENGTH:
        text = ' ' + text
    text = text[-SEQUENCE_LENGTH:]
    for i in range(words):
        text = text[-SEQUENCE_LENGTH:]
        pred = predict_completions(text, 1)[0]
        text = text + pred
        textOG = textOG + pred
        # print(text)
        pass
    return textOG

# print(genSentence("Today I will never", 10))

def padInput(text):
    while len(text) < SEQUENCE_LENGTH:
        text = ' ' + text
    text = text[-SEQUENCE_LENGTH:]
    return text

def NSMW(text, n, m):
    textOG = text
    text = text.lower()
    while len(text) < SEQUENCE_LENGTH:
        text = ' ' + text
    text = text[-SEQUENCE_LENGTH:]
    sentences = []
    x = prepare_input(text.lower())
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, top_n=m)
    for char in next_index:
        sentences.append(genSentence(textOG+indices_char[char], n))
    pass
    return sentences

@app.route('/predict')
def predict():
    c = request.args.get('a', 1, type=str)
    sentences = NSMW(c, 10, 10)
    print(sentences)
    return jsonify(result=sentences)

@app.route("/")
def index():
    return render_template('index.html')
app.run(host='0.0.0.0', port=50000, threaded=False)
