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
from keras.layers import CuDNNLSTM 
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

SEQUENCE_LENGTH = 80
step = 4
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
# model = load_model('D:/Predictive-Text/experiments/predictiveTrump/20200330-140314/weights/weights-improvement-04-0.6747.hdf5')

model = Sequential()

model.add(CuDNNLSTM(len(chars) * 5, input_shape=(SEQUENCE_LENGTH, len(chars))))
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
# reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=2, min_lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

models = [model]
args = [{'x':X,
         'y':y,
         'batch_size':124,
         'epochs':50,
         'shuffle':False,
         'validation_split':0.05,}]
         # 'callbacks': [reduce_lr]

ml = MLEXPS()
ml.setTopic('predictiveTrump')
ml.setCopyFileList(['train.py'])
ml.setModels(models)
ml.setArgList(args)
ml.saveBestOnly = False
ml.startExprQ()
