import os
import sys
import time
import keras
import heapq
import numpy as np
import tensorflow as tf
from shutil import copyfile
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, GRU, TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop

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
# speeches = 'data/trump/speeches/clean/cleanSpeech.txt'
tweets = 'data/trump/tweets/clean/cleanTweets.txt'
text = open(tweets, encoding="utf8").read().lower()
# text = text + open(speeches, encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SEQUENCE_LENGTH = 20
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

os.makedirs("models/Trump/" + str(timestr), exist_ok=True)
filepath = "models/Trump/" + str(timestr) + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
copyfile('main.py', "models/Trump/" + str(timestr) + "/main.py")
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=2, min_lr=0.00001)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# model = load_model('D:/Predictive-Text/models/Trump/20200324-181225/weights-improvement-01-0.6565.hdf5')
# model = load_model('D:/Predictive-Text/models/Trump/20200324-214657/weights-improvement-01-0.5648.hdf5')

# Making Model
model = Sequential()

model.add(GRU(len(chars) * 5, input_shape=(SEQUENCE_LENGTH, len(chars))))

model.add(Dense(len(chars) * 3))
model.add(Dense(len(chars) * 2))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

with open("models/Trump/" + str(timestr) + "/" + 'index.txt', 'w') as file:
    file.write('\ncorpus length: ' + str(len(text)))
    file.write(f'\nunique chars: {len(chars)}')
    file.write('\nSequence_Length: ' + str(SEQUENCE_LENGTH))
    file.write('\nStep: '+  str(step))
    # file.write(chars)
    # file.write(model.summary())

history = model.fit(X, y, validation_split=0.05, batch_size=124, epochs=6, shuffle=True, callbacks=[checkpoint,reduce_lr]).history
