from tensorflow import keras

from tensorflow.keras import backend as K
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import tensorflow_addons as tfa

# Test stuff

def class_precision_eval(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = tf.cast(K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1))), tf.float32)
    predicted_positives = tf.cast(K.sum(K.round(K.clip(class_y_pred, 0, 1))), tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def class_recall_eval(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = tf.cast(K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1))), tf.float32)
    possible_positives = tf.cast(K.sum(K.round(K.clip(class_y_true, 0, 1))), tf.float32)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def class_f1_eval(y_true, y_pred, class_num):
    precision = class_precision_eval(y_true, y_pred, class_num)
    recall = class_recall_eval(y_true, y_pred, class_num)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def macro_f1_eval(y_true, y_pred, num_classes):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    return K.sum([class_f1_eval(y_true, y_pred, i) for i in range(1, num_classes)]) / num_classes


def micro_recall_eval(y_true, y_pred):
    true_positives = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
    possible_positives = tf.cast(K.sum(K.round(K.clip(y_true, 0, 1))), tf.float32)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def micro_precision_eval(y_true, y_pred):
    true_positives = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
    predicted_positives = tf.cast(K.sum(K.round(K.clip(y_pred, 0, 1))), tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def micro_f1_eval(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    precision = micro_precision_eval(y_true, y_pred)
    recall = micro_recall_eval(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def macro_f1_evaluation(y_trues, y_preds, num_classes):
    f1_list = []
    e = 0.00001

    for c in range(1, num_classes):
        tps = []
        fps = []
        fns = []
        for y_true, y_pred in zip(y_trues, y_preds):
            tp = sum([1 for p, g in zip(y_pred, y_true) if p == g and p == c])
            fp = sum([1 for p, g in zip(y_pred, y_true) if p != g and p == c])
            fn = sum([1 for p, g in zip(y_pred, y_true) if p == 0 and g == c])
            
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        
        tps = sum(tps)
        fps = sum(fps)
        fns = sum(fns)

        p = tps / (tps + fps + e)
        r = tps / (tps + fns + e)
        f1 = 2 * p * r / (p + r + e)
        f1_list.append(f1)
    macro_f1 = sum(f1_list) / num_classes
    return macro_f1

def micro_f1_evaluation(y_trues, y_preds, num_classes):
    tps = []
    fps = []
    fns = []
    e = 0.00001


    # for y_true, y_pred in zip(y_trues, y_preds):
        # print("y_true:", y_true)
        # print("y_pred:", y_pred)
    for c in range(1, num_classes):
        for y_true, y_pred in zip(y_trues, y_preds):
            tp = sum([1 for p, g in zip(y_pred, y_true) if p == g and p == c])
            fp = sum([1 for p, g in zip(y_pred, y_true) if p != g and p == c])
            fn = sum([1 for p, g in zip(y_pred, y_true) if p == 0 and g == c])
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

#     print("micro eval stats:")
#     print("TPS:", tps)
#     print("FPS:", fps)
#     print("FNS:", fns)

    tps = sum(tps)
    fps = sum(fps)
    fns = sum(fns)
    micro_f1 = tps / (tps + 0.5 * (fps + fns) + e)
    return micro_f1

