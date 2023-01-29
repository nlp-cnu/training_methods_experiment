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

from utilities.DataGenerator import *
from utilities.constants import *


class Classifier:
    """
    Classifier class holds a language model and a classifier.
    Variables:
    self.language_mode_name - specifies which HuggingFace language model to use
    self.tokenizer - an instance of the tokenizer corresponding to the language model
    self.model - TF/Keras classification model. Model architecture is flexible, but 
        is set once the model is compiled

    Training uses a DataGenerator object that ensures batches are correctly divided.
    This object also allows for variable sized batches.
    """

    def __init__(self):
        """
        This is an abstract base class. In the constructor you must define:
        self.tokenizer
        self.model
        """
        self.tokenizer = None
        self.model = None


    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=MAX_EPOCHS, csv_log_file=None, early_stop_patience=None):
        """
        Trains the classifier
        :param x: the training data
        :param y: the training labels
        :param batch_size: the batch size
        :param validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
        :param epochs: the number of epochs to train for
        """
        training_data = DataGenerator(x, y, self.tokenizer, batch_size=batch_size)

        if validation_data is not None:
            validation_data = DataGenerator(validation_data[0], validation_data[1], self.tokenizer, batch_size=batch_size)

        callbacks = []
        if csv_log_file:
            csv_logger = tf.keras.callbacks.CSVLogger(csv_log_file, separator="\t", append=False)
            callbacks.append(csv_logger)

        if early_stop_patience:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_micro_f1', patience=early_stop_patience, mode='max') # , restore_best_weights) <== auto tracks model weights with best scores
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
            tokenized = self.tokenizer(list(x), padding=True, truncation=True, max_length=MAX_NUM_TOKENS, return_tensors='tf')
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
        dg = DataGenerator(X, y, self.tokenizer, batch_size=batch_size)
#        if not isinstance(X, tf.keras.utils.Sequence):
#            tokenized = self.tokenizer(list(X), padding=True, truncation=True, max_length=MAX_NUM_TOKENS, return_tensors='tf')
#            X = (tokenized['input_ids'], tokenized['attention_mask'])
        return self.model.evaluate(dg) # , verbose=SILENT)


class MultiClass_Token_Classifier(Classifier):

    """
    This constructor creates the model and compiles it for training
    """
    def __init__(self, language_model_name, num_classes):
        Classifier.__init__(self)
        self.language_model_name = language_model_name
        self.num_classes = num_classes

        # create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)

        # create the language model
        if os.path.isdir(language_model_name):
            language_model = TFBertModel.from_pretrained(self.language_model_name, from_pt=True, local_files_only=True)
        else:
            language_model = TFAutoModel.from_pretrained(self.language_model_name)
        language_model.trainable = True

        # Model takes the tokenizer input_ids and padding_mask
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        softmax_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        final_output = softmax_layer(embeddings)
        
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss=CATEGORICAL_CROSSENTROPY,
            metrics=[self.micro_f1, self.macro_f1]
        )


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


    def macro_f1(self, y_true, y_pred):
        return K.sum([self.class_f1(y_true, y_pred, i) for i in range(self.num_classes)]) / self.num_classes


    def micro_recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def micro_precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def micro_f1(self, y_true, y_pred):
        precision = self.micro_precision(y_true, y_pred)
        recall = self.micro_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


