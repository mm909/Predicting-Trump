from keras.models import load_model
import numpy as np
import heapq

# model = load_model('models/Trump/20200318-162752/weights-improvement-08-0.5838.hdf5')
model = load_model('D:/Predictive-Text/experiments/predictiveTrump/20200327-095125/weights/weights-improvement-04-0.6693.hdf5')

SEQUENCE_LENGTH = 40

# Gettings data
speeches = 'data/trump/speeches/clean/cleanSpeech.txt'
tweets = 'data/trump/tweets/clean/cleanTweets.txt'
text = open(tweets, encoding="utf8").read().lower()
text = text + open(speeches, encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    print(preds)
    for index in heapq.nlargest(5, range(len(preds)), preds.take):
        print(indices_char[index], preds[index])
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



# text = 'China '
# seq = text[:20].lower()
# while len(seq) < 20:
#     seq = ' ' + seq
# print(seq)
# print(predict_completions(seq, 1))
# print()

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
        print(text)
        pass
    return textOG

print(genSentence("Today I ", 1))
