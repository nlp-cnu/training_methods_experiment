import os
import csv
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from transformers import AutoTokenizer, TFAutoModel

from utilities.constants import *

import regex
import re

class Token_Classification_Dataset:
    def __init__(self, data_file_path, num_classes, tokenizer, seed=SEED, test_set_size=0,
                 max_num_tokens=512, shuffle_data=True):
        self.seed = seed
        self.num_classes = num_classes
        self.tokenizer = tokenizer

        # load and preprocess the data
        df = self.preprocess(data_file_path)
        self.data = df["text"].tolist()

        # create the labels datastructure from the loaded labels in the data frame
        # self.labels = np.zeros([len(self.df['annotation']), max_num_tokens, self.num_classes])
        self.labels = []

        # Convert from categorical encoding to binary encoding
        # Need to make a big array that is i x j x num_classes, where i is the ith token, j is the number of tokens
        num_lost = 0
        num_samples = len(df['annotation'])
        for sample_num in range(num_samples):
            num_tokens = len(df['annotation'][sample_num])

            # check if the annotations are getting truncated
            if num_tokens > max_num_tokens:
                num_lost += num_tokens - max_num_tokens

            # create a matrix of annotations for this line. That is, vector per token in the line
            #  up to the max_num_tokens
            # for j in range(num_tokens)[:max_num_tokens]:
            sample_annotations = np.zeros([num_tokens, num_classes])
            for token_num in range(num_tokens):
                # grab the class the token belongs to
                true_class = int(df['annotation'][sample_num][token_num])

                # create the vector for this annotation
                if num_classes > 1:
                    # self.labels[i][j][int(true_class)] = 1.0
                    sample_annotations[token_num, true_class] = 1.0
                else:  # binary
                    # 0 indicates the None class, which we don't annotate, otherwise set the class to 1
                    if true_class > 0:
                        class_index = true_class - 1
                        # self.labels[i][j][int(class_index)] = 1.0
                        sample_annotations[token_num, class_index] = 1.0

            # add this sample (line) to the list of annotations
            self.labels.append(sample_annotations)

        print("Number of lost tokens due to truncation:", num_lost)

    def preprocess(self, input_file):
        # Want to grab the training data, expand all the labels using the tokenizer
        # Creates new label that accounts for the tokenization of a sample
        def tokenize_sample(df_sample):
            # get a list containing space separated tokens
            tokens = df_sample['text'].split(' ')

            # get the length of each token
            token_lengths = []
            for token in tokens:
                tokenized = self.tokenizer(token, return_tensors='tf')
                length = len(tokenized['input_ids'][0])
                length -= 2 # remove CLS and SEP
                token_lengths.append(length)

            # Create the new labels, which maps the space separated labels to token labels
            new_labels = []
            # add a 0 label for the [CLS] token
            new_labels.append(0)
            # extend each label to the number of tokens in that space separated "word"
            labels = df_sample['annotation']
            for i in range(len(labels)):
                # add the new labels
                labels_for_this_word = [labels[i]] * token_lengths[i]
                new_labels.extend(labels_for_this_word)
            # add a 0 label for the SEP token and return
            new_labels.append(0)

            # check to make sure the lengths match (unnecessary, but useful for debugging)
            #tokenized = self.tokenizer(sample['text'], return_tensors='tf')
            #if(len(tokenized['input_ids'][0]) != len(new_labels)):
            #    print(f"MISMATCH: {len(tokenized['input_ids'][0])}, {len(new_labels)}, {tokenized['input_ids']}, {sample['text']}")
            #else:
            #    print("MATCHED")

            df_sample['annotation'] = new_labels
            return df_sample

        # assumes classes are encoded as a real number, so a single annotation per class
        df = pd.read_csv(input_file, delimiter='\t', header=None, names=['text', 'annotation'], keep_default_na=False,
                         quoting=csv.QUOTE_NONE)  # , encoding='utf-8')

        # replace non-standard space characters with a space
        df['text'] = df['text'].apply(lambda x: regex.sub(r'\p{Zs}', ' ', x))

        # add spaces between all 'naive' tokens which are the tokens with labels. This ensures the tokenizer
        # will be equal to or longer than the number of labels (important for cases like "..." which contain 3
        # labels (from pre-processing) but may only be treated as a single token
        df['text'] = df['text'].apply(lambda x: ' '.join(re.findall(r'\b\w+\b|[^\s\w]', x)))

        # NOTE: This could make performance worse, but [UNK] tokens are a big problems for converting between formats
        # replace non-ascii characters with *
        #  if we just remove them then it can throw off the labels
        for i in range(len(df['text'].values)):
            text_list = list(df.iloc[i]['text'])
            for j, char in enumerate(text_list):
                if ord(char) > 127:
                    # replace everything else with an asterisk
                    text_list[j] = '*'
            df.iloc[i]['text'] = "".join(text_list)

        # convert the annotation to numbers
        df['annotation'] = df['annotation'].apply(literal_eval)
        # expand the annotations to match the tokens (a word may be multiple tokens)
        df = df.apply(tokenize_sample, axis=1)

        # return the processed dataframe
        return df

    def get_folds(self, k):
        kf = KFold(n_splits=k, random_state=self.seed, shuffle=True)
        return kf.split(self.data)

