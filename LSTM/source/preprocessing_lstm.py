import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from torchtext.vocab import FastText

import torch

class Preprocessing:

    def __init__(self, args):

        # Initializing and storing the data and data parameter

        self.data = 'data/train_subtask1.csv'
        self.max_len = args.max_len
        self.max_words = args.max_words
        self.test_size = args.test_size
        self.embedding=FastText('simple')

    def load_data(self):

        # loading the stored data and removing uneseccary columns for the task

        df = pd.read_csv(self.data)
        df.drop(['sentence'], axis=1, inplace=True)

        X = df['text'].values
        Y = df['_id'].values

        # train and test splitting of the data

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=self.test_size)

    def prepare_tokens(self):

        # Updateing internal vocabulary based on the sequences.

        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)

    def sequence_to_token(self, x):

        # Converting the sequence to tokens

        sequences = self.tokens.embedding(torch.tensor[x])
        return sequence.pad_sequences(sequences, maxlen=self.max_len)
