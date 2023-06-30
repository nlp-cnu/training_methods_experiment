import os

from tensorflow.keras import backend as K
from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import tensorflow as tf

tf.get_logger().setLevel("WARNING")
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa
from tensorflow import keras

import numpy as np

from utilities.DataHandler import *
from utilities.constants import *

import re

class MultiClass_Token_Classifier:
    """
    This constructor creates the model and compiles it for training
    """

    def __init__(self, language_model_name, num_classes, tokenizer, max_num_tokens):
        self.language_model_name = language_model_name
        self.max_num_tokens = max_num_tokens
        self.num_classes = num_classes
        self.tokenizer = tokenizer

        # create the language model
        if os.path.isdir(language_model_name):
            if "ONTO" in language_model_name or "INTER" in language_model_name:
                self.language_model = TFBertModel.from_pretrained(self.language_model_name, local_files_only=True)
            else:
                self.language_model = TFBertModel.from_pretrained(self.language_model_name, from_pt=True,
                                                                  local_files_only=True)
        else:
            self.language_model = TFAutoModel.from_pretrained(self.language_model_name)

        self.language_model.trainable = True

        # Model takes the tokenizer input_ids and padding_mask
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = self.language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # create the output layer
        if num_classes == 1: # binary
            activation = 'sigmoid'
            loss_function = 'binary_crossentropy'
        else: # multi-label
            activation = 'softmax'
            loss_function = 'categorical_crossentropy'


        output_layer = tf.keras.layers.Dense(self.num_classes, activation=activation)
        final_output = output_layer(embeddings)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        metrics = [self.micro_f1, self.macro_f1]
        self.model.compile(
            optimizer=optimizer,
            loss=CATEGORICAL_CROSSENTROPY,
            metrics=[self.micro_f1, self.macro_f1]
        )

    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=MAX_EPOCHS, csv_log_file=None,
              early_stop_patience=None, restore_best_weights=True):
        """
        Trains the classifier
        :param x: the training data
        :param y: the training labels
        :param batch_size: the batch size
        :param validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
        :param epochs: the number of epochs to train for
        """
        training_data = DataHandler(x, y, self.tokenizer, self.num_classes, batch_size=batch_size, max_num_tokens=self.max_num_tokens)

        if validation_data is not None:
            validation_data = DataHandler(validation_data[0], validation_data[1], self.tokenizer, self.num_classes,
                                            batch_size=batch_size, max_num_tokens=self.max_num_tokens)

        callbacks = []
        if csv_log_file:
            csv_logger = tf.keras.callbacks.CSVLogger(csv_log_file, separator="\t", append=False)
            callbacks.append(csv_logger)

        if early_stop_patience:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_micro_f1', patience=early_stop_patience,
                                                          mode='max',
                                                          restore_best_weights=restore_best_weights)  # , restore_best_weights) <== auto tracks model weights with best scores
            callbacks.append(early_stop)

        return self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            verbose=SILENT,
            callbacks=callbacks
        )

    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :param batch_size: batch size
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(list(x), padding=True, truncation=True, max_length=self.max_num_tokens,
                                       return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])
        return self.model.predict(x, batch_size=batch_size, verbose=SILENT)

    def evaluate(self, X, y, batch_size=BATCH_SIZE):
        """
        Evaluate testset for data
        :param X: data
        :param y: labels
        :param batch_size: batch size
        :return: tf.History object
        """
        dg = DataHandler(X, y, self.tokenizer, self.num_classes, batch_size=batch_size, max_num_tokens=self.max_num_tokens)
        #        if not isinstance(X, tf.keras.utils.Sequence):
        #            tokenized = self.tokenizer(list(X), padding=True, truncation=True, max_length=self.max_num_tokens, return_tensors='tf')
        #            X = (tokenized['input_ids'], tokenized['attention_mask'])
        return self.model.evaluate(dg)  # , verbose=SILENT)

    def save_language_model(self, directory_path):
        self.language_model.save_pretrained(directory_path)

    def class_precision(self, y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def class_recall(self, y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def class_f1(self, y_true, y_pred, class_num):
        precision = self.class_precision(y_true, y_pred, class_num)
        recall = self.class_recall(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    # range starts at 1 because we don't want to include the None Class
    def macro_f1(self, y_true, y_pred):
        return K.sum([self.class_f1(y_true, y_pred, i) for i in range(1, self.num_classes)]) / self.num_classes

    # 1: to exclude the none class
    def micro_recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[1:] * y_pred[1:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[1:], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    #1: to exlcude the none class
    def micro_precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[1:] * y_pred[1:], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[1:], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def micro_f1(self, y_true, y_pred):
        precision = self.micro_precision(y_true, y_pred)
        recall = self.micro_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
