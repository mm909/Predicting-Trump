import os
import sys
import time
import keras
import heapq
import numpy as np
import tensorflow as tf
from shutil import copyfile
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, GRU, TimeDistributed, BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from MLEXPS.MLEXPS import *

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    return x

def temperatureSample(preds, temperature=1.0):
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


timestr = time.strftime("%Y%m%d-%H%M%S")

# Gettings data
speeches = 'data/trump/speeches/clean/cleanSpeech.txt'
tweets = 'data/trump/tweets/clean/cleanTweets.txt'
text = open(tweets, encoding="utf8").read().lower()
text = text + open(speeches, encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SEQUENCE_LENGTH = 25
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print("X.shape:", X.shape)
print("y.shape:", y.shape)

# Making Model
model = Sequential()

model.add(GRU(len(chars) * 5, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Dense(len(chars) * 2))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Dense(len(chars) * 2))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

models = [model]
args = [{'x':X,
         'y':y,
         'batch_size':124,
         'epochs':4,
         'shuffle':True,
         'validation_split':0.05}]

ml = MLEXPS()
ml.setTopic('predictiveTrump')
ml.setCopyFileList(['train.py'])
ml.setModels(models)
ml.setArgList(args)
ml.startExprQ()

# history = model.fit(X, y, validation_split=0.05, batch_size=100, epochs=4, shuffle=True, callbacks=[checkpoint,reduce_lr]).history

'''
|- 20200324-204855:        GRU Base (D = Tweets v1, Github, FactBase(3/23)) - 0.6619
   |- 20200325-014254:     GRU Base (Continued)  (D = Tweets v1, Github, FactBase(3/23)) - 0.66178
   |- 20200325-164139:     GRU Base w/ BN & SELU (D = Tweets v2, Github, FactBase(3/24)) - 0 (RUNNING)
     |- PLANNED:           GRU Base w/ BN & SELU Batch = 148 (D = Tweets v2, Github, FactBase(3/24)) - 0
     |- PLANNED:           GRU Base w/ BN & SELU Batch = 100 (D = Tweets v2, Github, FactBase(3/24)) - 0
   |- 20200325-102310:     GRU Base w/ SELU & BN (D = Tweets v1) - 0.59794
     |- 20200325-113640:   GRU Base w/ SELU & BN (D = Tweets v1, Github, FactBase(3/24)) - 0.6711
       |- 20200325-145253: GRU Base w/ SELU & BN Not Shuffled (D = Tweets v1, Github, FactBase(3/24)) - 0.6272
   |- 20200324-220128:     GRU Base - Tweets Only (D = Tweets v1) - 0.59347
      |- 20200324-225921:  GRU Base - Wider (D = Tweets v1) - 0.59217
      |- 20200325-081236:  GRU Base - Deeper (D = Tweets v1) - 0.59403
      |- 20200325-091649:  GRU Base w/ SELU (D = Tweets v1) - 0.59142
'''
