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
        if self.num_classes == 1: # binary
            activation = 'sigmoid'
            loss_function = 'binary_crossentropy'
        else: # multi-label
            activation = 'softmax'
            loss_function = 'categorical_crossentropy'


        output_layer = tf.keras.layers.Dense(self.num_classes, activation=activation)
        final_output = output_layer(embeddings)

        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # set up the metrics
        if self.num_classes == 1: #binary
            metrics = [self.micro_f1_binary_multilabel, self.macro_f1_binary_multilabel]
        else: #multiclass
            metrics = [self.micro_f1_multiclass, self.macro_f1_multiclass]

        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
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
            if self.num_classes == 1:
                metric_to_monitor='val_micro_f1_binary_multilabel'
            else:
                metric_to_monitor='val_micro_f1_multiclass'
                
            early_stop = tf.keras.callbacks.EarlyStopping(monitor=metric_to_monitor, patience=early_stop_patience,
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


    ##########
    ### Multiclass Metrics (excludes the 0th = None class)
    # range starts at 1 because we don't want to include the None Class
    def macro_f1_multiclass(self, y_true, y_pred):
        return K.sum([self.class_f1_multiclass(y_true, y_pred, i) for i in range(1, self.num_classes)]) / self.num_classes

    def micro_f1_multiclass(self, y_true, y_pred):
        precision = self.micro_precision_multiclass(y_true, y_pred)
        recall = self.micro_recall_multiclass(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    # 1: to exclude the none class
    def micro_recall_multiclass(self, y_true, y_pred):
        #true_positives = K.sum(K.round(K.clip(y_true[1:] * y_pred[1:], 0, 1)))
        #possible_positives = K.sum(K.round(K.clip(y_true[1:], 0, 1)))
        #old_recall = true_positives / (possible_positives + K.epsilon())
        #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        #possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        #old_recall = true_positives / (possible_positives + K.epsilon())
        #return recall

        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true negatives are correctly predicted as None ()
        #tn = K.sum(predictions == 0 and golds == 0)
        #tn = K.sum(tf.cast(predictions == 0, tf.int32) * tf.cast(golds == 0, tf.int32))

        # predicted as anything but the correct class (since it was missed for that class) for classes that aren't None (0) only
        #fn = K.sum(predictions != golds and golds != 0)
        fn = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(golds != 0, tf.float32))
        
        # true positive are the correctly predicted class but excluding the None (0) class
        #tp = K.sum(predictions == golds and predictions != 0)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(predictions != 0, tf.float32))
               
        # predicted as anything but the correct class (since it was predicted as some other class), but not predicted as None (0) 
        #fp = K.sum(predictions != golds and predictions != 0)
        #fp = K.sum(tf.cast(predictions != golds, tf.int32) * tf.cast(predictions != 0, tf.int32))

        recall = tp / (tp + fn + K.epsilon())
        return recall        
    
    
    #1: to exlcude the none class --- this definately includes the none class now
    def micro_precision_multiclass(self, y_true, y_pred):
        #true_positives = K.sum(K.round(K.clip(y_true[1:] * y_pred[1:], 0, 1)))
        #predicted_positives = K.sum(K.round(K.clip(y_pred[1:], 0, 1)))
        #old_precision = true_positives / (predicted_positives + K.epsilon())
        #return precision
        
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true negatives are correctly predicted as None ()
        #tn = K.sum(predictions == 0 and golds == 0)
        #tn = K.sum(tf.cast(predictions == 0, tf.int32) * tf.cast(golds == 0, tf.int32))

        # predicted as anything but the correct class (since it was missed for that class) for classes that aren't None (0) only
        #fn = K.sum(predictions != golds and golds != 0)
        #fn = K.sum(tf.cast(predictions != golds, tf.int32) * tf.cast(golds != 0, tf.int32))
        
        # true positive are the correctly predicted class but excluding the None (0) class
        #tp = K.sum(predictions == golds and predictions != 0)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(predictions != 0, tf.float32))
               
        # predicted as anything but the correct class (since it was predicted as some other class), but not predicted as None (0) 
        #fp = K.sum(predictions != golds and predictions != 0)
        fp = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(predictions != 0, tf.float32))

        precision = tp / (tp + fp + K.epsilon())
        return precision    


    def class_precision_multiclass(self, y_true, y_pred, class_num):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true positive are when predicted = gold (for the class_num)
        #tp = K.sum(predictions == golds and gold == class_num)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
               
        # false positives are when things are predicted as this class that aren't
        #fp = K.sum(predictions != golds and predictions == class_num)
        fp = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(predictions == class_num, tf.float32))

        precision = tp / (tp + fp + K.epsilon())
        return precision
        
    
    def class_recall_multiclass(self, y_true, y_pred, class_num):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true positive are when predicted = gold (for the class_num)
        #tp = K.sum(predictions == golds and gold == class_num)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
        
        # a sample of class_num that wasn't classified as class_num
        #fn = K.sum(predictions != golds and golds == class_num)
        fn = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
                      
        recall = tp / (tp + fn + K.epsilon())
        return recall


    def class_f1_multiclass(self, y_true, y_pred, class_num):
        precision = self.class_precision_multiclass(y_true, y_pred, class_num)
        recall = self.class_recall_multiclass(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    

    #########
    ### Binary/multilabel metrics (includes the 0th class)
    def macro_f1_binary_multilabel(self, y_true, y_pred):
        #for i in range(self.num_classes):
        #    tf.print(y_true)
        #    tf.print(y_pred)
        #    tf.print("i = ")
        #    tf.print(i)
        #    tf.print("class_f1 = ")
        #    tf.print(self.class_f1_binary_multilabel(y_true, y_pred,i))
        #    tf.print("")
        #    tf.print(self.num_classes)
        #    tf.print(K.sum([self.class_f1_binary_multilabel(y_true, y_pred, i) for i in range(self.num_classes)]) / self.num_classes)
        #    tf.print(K.sum([self.class_f1_binary_multilabel(y_true, y_pred, 0)]))
        #    tf.print("")
        
        return K.sum([self.class_f1_binary_multilabel(y_true, y_pred, i) for i in range(self.num_classes)]) / self.num_classes

    def micro_f1_binary_multilabel(self, y_true, y_pred):
        precision = self.micro_precision_binary_multilabel(y_true, y_pred)
        recall = self.micro_recall_binary_multilabel(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    def micro_recall_binary_multilabel(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def micro_precision_binary_multilabel(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def class_precision_binary_multilabel(self, y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def class_recall_binary_multilabel(self, y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def class_f1_binary_multilabel(self, y_true, y_pred, class_num):
        precision = self.class_precision_binary_multilabel(y_true, y_pred, class_num)
        recall = self.class_recall_binary_multilabel(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
