from tensorflow import keras

from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model), do they work for multi-class problems too?
def micro_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def micro_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def micro_f1(y_true, y_pred):
    precision = micro_precision(y_true, y_pred)
    recall = micro_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def class_precision(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def class_recall(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def class_f1(y_true, y_pred, class_num):
    precision = class_precision(y_true, y_pred, class_num)
    recall = class_recall(y_true, y_pred, class_num)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def macro_precision(y_true, y_pred): #, num_classes):
    num_classes = 6
    return K.sum([class_precision(y_true, y_pred, i) for i in range(num_classes)]) / num_classes


def macro_recall(y_true, y_pred): #, num_classes):
    num_classes = 6
    return K.sum([class_recall(y_true, y_pred, i) for i in range(num_classes)]) / num_classes


def macro_f1(y_true, y_pred): #, num_classes):
    num_classes = 6
    return K.sum([class_f1(y_true, y_pred, i) for i in range(num_classes)]) / num_classes

