import os
import csv
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from transformers import AutoTokenizer, TFAutoModel

from utilities.constants import *

class Dataset:
    def __init__(self, seed=SEED, test_set_size=0):
        self.seed = seed
        self.test_set_size = test_set_size

    def _test_train_split(self, data, labels):
        self.train_X = data
        self.train_Y = labels
        self.test_X = None
        self.test_Y = None
           
    def _determine_class_weights(self):
        # determine class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_Y),
            y=self.train_Y 
        )
        self.class_weights = dict(enumerate(self.class_weights))

    def get_train_data(self):
        return self.train_X, self.train_Y

    def get_train_class_weights(self):
        return self.class_weights

    def get_test_data(self):
        if self.test_X is None or self.test_Y is None:
            raise Exception("Error: test data does not exist")
        return self.test_X, self.test_Y

    def preprocess_data(self, data):
        return data.tolist()
    

class Token_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, num_classes, language_model_name, seed=SEED, test_set_size=0):
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)
        self.num_classes = num_classes
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.df = self.preprocess(data_file_path, tokenizer)

        self.labels = np.zeros([len(self.df['annotation']), MAX_NUM_TOKENS, num_classes])
        # Need to make a big array that is ixjxnum_classes, where i is the ith token, j is the number of tokens
        num_lost = 0

        num_samples = len(self.df['annotation'])
        for i in range(num_samples):
            num_tokens = len(self.df['annotation'][i])
            if num_tokens > 512:
                num_lost += num_tokens - 512
            for j in range(num_tokens)[:MAX_NUM_TOKENS]:
                positive_class_index = self.df['annotation'][i][j]
                self.labels[i][j][int(positive_class_index)] = 1.0

        self.data = self.df["text"].tolist()
        self._test_train_split(self.data, self.labels)
        print("Number of lost tokens:", num_lost)
        

    def preprocess(self, input_file, tokenizer):
        # Want to grab the training data, expand all the labels using the tokenizer
        # Shuffle the samples, save to new file that will be called during training
        
        # Creates new label that accounts for the tokenization of a sample
        def tokenize_sample(sample, tokenizer):
            new_label = []
            tok_lengths = [len(tok) - 2 for tok in tokenizer(sample['text'].split())['input_ids']]

            label = sample['annotation']
            # this index is following the labels, but the tok_lengths index
            # is one less because it does not account for the [CLS] and [SEP] tags
            for i in range(len(label)):
                l = label[i]
                new_l = [l] * tok_lengths[i]
                new_label.extend(new_l)
            new_label = [0] + new_label + [0]
            return new_label
        

        df = pd.read_csv(input_file, delimiter='\t', header=None, names=['text', 'annotation'], keep_default_na=False, quoting=csv.QUOTE_NONE)
        df = df.sample(frac=1, random_state=SEED)
        df['annotation'] = df['annotation'].apply(literal_eval)
        df['annotation'] = df.apply(tokenize_sample, tokenizer=tokenizer, axis=1)

        # See if you can just return this new dataframe, instead of saving all of this extra data
        return df

    def get_folds(self, k):
        kf = KFold(n_splits=k)
        return kf.split(self.df)

