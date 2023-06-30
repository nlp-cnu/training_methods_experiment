import tensorflow as tf

tf.get_logger().setLevel("WARNING")
import pandas as pd
import numpy as np

from utilities.constants import *


class DataHandler(tf.keras.utils.Sequence):
    """
    Class to generate batches.
    The datagenerator inherits from the sequence class which is used to generate
    data for each batch of training. Using a sequence generator allows for
    variable size batches. (depending on the maximum length of sequences in the batch)
    """

    def __init__(self, x_set, y_set, tokenizer, num_classes, batch_size=BATCH_SIZE, shuffle=True, max_num_tokens=MAX_NUM_TOKENS):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #Tokenize the input
        tokenized = self.tokenizer(batch_x, padding=True, truncation=True, max_length=self.max_num_tokens, return_tensors='tf')

        #trim the y_labels to be the max length of the batch
        num_samples = tokenized['input_ids'].shape[0]
        num_batch_tokens = tokenized['input_ids'].shape[1]
        extended_batch_y = np.zeros([num_samples, num_batch_tokens, self.num_classes])
        for i, sample in enumerate(batch_y):
            # The sample is a num_tokens x num_labels matrix
            num_tokens = sample.shape[0]\

            # crop the labels if necessary
            if sample.shape[1] >= self.max_num_tokens:
                sample = sample[:self.max_num_tokens, :]

            extended_batch_y[i, :num_tokens, :] = sample[:, :]

        return (tokenized['input_ids'], tokenized['attention_mask']), extended_batch_y

    def on_epoch_end(self):
        """
        Method is called each time an epoch ends. This will shuffle the data at
        the end of an epoch, which ensures the batches are not identical each epoch
        :return:
        """
        if self.shuffle:
            idxs = np.arange(len(self.x))
            np.random.shuffle(idxs)
            self.x = [self.x[idx] for idx in idxs]
            self.y = [self.y[idx] for idx in idxs]

