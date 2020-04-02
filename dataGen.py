import keras
import tracemalloc
import numpy as np
from sys import getsizeof
import math

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
        filename,
        batch_size = 128,
        step_size = 1,
        sequence_length = 20,
        unique_chars = 56,
        batch_limit = 10000):

        print('DataGenerator v1')
        self.filename   = filename
        self.batch_size = batch_size
        self.step_size  = step_size
        self.sequence_length = sequence_length
        self.unique_chars = unique_chars
        self.batch_limit  = batch_limit

        self.file        = open(self.filename, 'r')
        self.text        = self.readfile()
        self.text_len    = len(self.text)

        self.chars        = None
        self.unique_chars = None
        self.char_indices = None
        self.indices_char = None
        self.createDict()

        self.batch_X = None
        self.batch_y = None
        self.batch_index = 0
        self.num_batches = self.__len__()
        self.batch_return = -1
        self.created_batches = 0

        print(self.__len__())
        print("batch_limit     :", self.batch_limit, "batches.")
        print("batch_size      :", self.batch_size, "examples.")
        print("step_size       :", self.step_size, "characters.")
        print("unique_chars    :", self.unique_chars, "characters.")
        print("sequence_length :", self.sequence_length, "characters.")

    def gen_full_batch(self):
        self.batch_X = np.zeros((self.batch_limit, self.batch_size, self.sequence_length, self.unique_chars))
        self.batch_y  = np.zeros((self.batch_limit, self.batch_size, self.unique_chars))
        eof = False
        start = 0
        batch_index_temp = 0
        self.created_batches = 0
        for batch in range(self.batch_limit):
            self.created_batches += 1
            for example in range(self.batch_size):
                start = self.batch_index + batch * self.batch_size * self.step_size
                end = start + example*self.step_size + self.sequence_length
                if end == len(self.text):
                     eof = True
                     break
                # print(self.text[start + example*self.step_size: start + example*self.step_size + self.sequence_length])
                keysX = [self.char_indices[value] for value in self.text[start + example*self.step_size: start + example*self.step_size + self.sequence_length]]
                keysy = [self.char_indices[value] for value in self.text[start + example*self.step_size + self.sequence_length]][0]
                for charIndex, key in enumerate(keysX):
                    self.batch_X[batch, example, charIndex, key] = 1
                self.batch_y[batch, example, key] = 1
                batch_index_temp += 1
            if eof:
                print('End of file')
                break
        self.batch_index += batch_index_temp

    def createDict(self):
        self.chars = sorted(list(set(self.text)))
        self.unique_chars = len(self.chars)
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        return

    def readfile(self):
        return self.file.read()

    # Must implement
    '''Number of batches per epoch'''
    def __len__(self):
        return int(((len(self.text) - self.sequence_length + 1 )/ (self.step_size) / self.batch_size))

    # Must implement
    # return a batch
    def __getitem__(self, index):
        # print('index', index)
        'Generate one batch of data'
        if self.batch_return == self.created_batches-1:
            self.batch_return = -1
        if self.batch_return == -1:
            print('Reset Batches')
            self.gen_full_batch()
            self.batch_return = 0
        X = self.batch_X[self.batch_return]
        y = self.batch_y[self.batch_return]
        self.batch_return += 1
        # print('Getting 1 batch')
        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        return

    # Optional
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def printMem(self):
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB;")
        return
