import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, GRU
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
import time
from shutil import copyfile
import os

timestr = time.strftime("%Y%m%d-%H%M%S")

# Gettings data
tweets = 'data/trump/tweets/trumpClean2.txt'
speeches = 'data/trump/speeches/textClean.txt'
text = open(tweets, encoding="utf8").read().lower()
text = text + open(speeches, encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SEQUENCE_LENGTH = 60
step = 3
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

print('')
print(sentences[200])
print(next_chars[200])
print('')
print("X.shape:", X.shape)
print("y.shape:", y.shape)
# Gettings data

os.makedirs("models/Trump/" + str(timestr), exist_ok=True)
filepath = "models/Trump/" + str(timestr) + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
copyfile('main.py', "models/Trump/" + str(timestr) + "/main.py")
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=2, min_lr=0.00001)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Making Model
model = Sequential()
model.add(GRU(len(chars), input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=124, epochs=10, shuffle=True, callbacks=[checkpoint,reduce_lr]).history

model.save('keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    return x

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

# quotes = [
#     "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
#     "That which does not kill us makes us stronger.",
#     "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
#     "And those who were seen dancing were thought to be insane by those who could not hear the music.",
#     "It is hard enough to remember my opinions, without also remembering my reasons for them!"
# ]

# for q in quotes:
#     seq = q[:40].lower()
#     print(seq)
#     print(predict_completions(seq, 5))
#     print()
