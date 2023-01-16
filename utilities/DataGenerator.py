import tensorflow as tf
tf.get_logger().setLevel("WARNING")
import pandas as pd
import numpy as np

from utilities.constants import *

class DataGenerator(tf.keras.utils.Sequence):
    """
    Class to generate batches.
    The datagenerator inherits from the sequence class which is used to generate
    data for each batch of training. Using a sequence generator allows for 
    variable size batches. (depending on the maximum length of sequences in the batch) 
    """
    
    def __init__(self, x_set, y_set, tokenizer, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = BATCH_SIZE
        print("BATCH_SIZE=", BATCH_SIZE)
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def __len__(self):
        print("len:", int(np.ceil(len(self.x) / self.batch_size)))
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = list(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        tokenized = self.tokenizer(batch_x, padding=True, truncation=True, max_length=MAX_NUM_TOKENS, return_tensors='tf')

        num_samples = tokenized['input_ids'].shape[0]
        num_tokens = tokenized['input_ids'].shape[1]
        num_classes = batch_y.shape[2]

        cropped_batch_y = np.zeros([num_samples, num_tokens, num_classes])
        for i in range(num_samples):
            cropped_batch_y[i][:][:] = batch_y[i][:num_tokens][:]
            
        return (tokenized['input_ids'], tokenized['attention_mask']), cropped_batch_y

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
            self.y = self.y[idxs]

