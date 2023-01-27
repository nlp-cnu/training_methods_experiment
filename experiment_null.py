"""
Code to convert the datasets likely sucks,
so we will validate it here
"""

import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import logging

from utilities.Classifier import *
from utilities.Dataset import *
from utilities.constants import *

def dataset_test(): 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    logging.set_verbosity("ERROR")
    for dataset_path in DOMAIN_SPECIFIC_DATASETS[0:1]:
        dataset_name = dataset_path.split(os.sep)[-1]
        print("Dataset:", dataset_name)
        class_map = DATASET_TO_CLASS_MAP[dataset_name]
        num_classes = len(class_map)
        print("Class mapping:", class_map)
        for language_model in ALL_MODELS[2:3]:
            language_model_name = language_model.split(os.sep)[-1]
            print("\tLanguage model:" + language_model_name)
            
            training_file_path = os.path.join(dataset_path, CONVERTED_DATASET_FILE)
            
            data = Token_Classification_Dataset(training_file_path, num_classes, language_model, seed=SEED)
            # folds = list(data.get_folds(NUM_FOLDS))
            folds = list(data.get_folds(2))

            test_micro_f1 = []
            test_macro_f1 = []


            predictions = []
            test_samples = []
            golds = []

            for index, train_test in enumerate(folds):
                train_index, test_index = train_test
                train_data = np.array(data.data)[train_index]
                train_labels = np.array(data.labels)[train_index]
                test_data = np.array(data.data)[test_index]
                test_labels = np.array(data.labels)[test_index]

                train_data_, val_data, train_labels_, val_labels = train_test_split(train_data, train_labels, test_size=VALIDATION_SIZE, random_state=3)

                classifier = MultiClass_Token_Classifier(language_model, num_classes)
                validation_metrics = classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels), early_stop_patience=EARLY_STOPPING_PATIENCE)
                validation_history = validation_metrics.history
                print(f"Validation history fold_{index}:", validation_history)
                target_metric = validation_history['val_micro_f1']
                
                num_epochs = target_metric.index(max(target_metric)) + 1
                
                classifier = MultiClass_Token_Classifier(language_model, num_classes)
                print(f"Test history fold_{index}:", classifier.train(train_data, train_labels, epochs=num_epochs).history)

                ps = classifier.predict(test_data)
                predictions.append(classifier.predict(test_data))
                golds.append(test_labels)

                # ^^^^^^^^^^^^^^

                evaluation = classifier.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)

                test_micro_f1.append(evaluation[1])
                test_macro_f1.append(evaluation[2])
                print()

            print("Model test scores using evaluate:")
            print("Test_micro_f1 across all folds:", test_micro_f1)
            print(f"Test_micro_f1 average score={np.mean(test_micro_f1)}, standard_deviation={np.std(test_micro_f1)}")
            print("Test_macro_f1 across all folds:", test_macro_f1)
            print(f"Test_macro_f1 average score={np.mean(test_macro_f1)}, standard_deviation={np.std(test_macro_f1)}")

            pred_micro_f1s = []
            pred_macro_f1s = []

            print()
            print("Model test scores using predict:")

            # For each fold there is a y_true and y_pred
            for p, g in zip(predictions, golds):
                # making y_pred and y_true have the same size by trimming
                num_samples = p.shape[0]
                max_num_tokens_in_batch = p.shape[1]
                # Transforms g to the same size as P and removes the NONE class
                # g = g[:, :max_num_tokens_in_batch, 1:]
                g = g[:, :max_num_tokens_in_batch, :]
                # removes the NONE class
                predictions = []
                for sample_pred, sample_gt, i in zip(p, g, range(num_samples)):
                    sample_pred = (sample_pred == sample_pred.max(axis=1)[:,None]).astype(int)
                    p[i] = sample_pred
                
                # p = p[:, :, 1:]

                num_classes_remaining = p.shape[-1]
                p = p.reshape((-1, num_classes_remaining))
                g = g.reshape((-1, num_classes_remaining))

                print("Report before none class:") 
                print("P before:", p.shape)
                print("G before:", g.shape)
                print("class 1234 predicts:", np.sum(p, axis=0))
                print("class 1234 ground truth:", np.sum(g, axis=0))

                print(classification_report(g, p, target_names=["none", "problem", "treatment", "test"]))

                g = g[:, 1:]
                p = p[:, 1:]

                print("Report after none class:")
                print("P after:", p.shape)
                print("G after:", g.shape)
                print("class 1234 predicts:", np.sum(p, axis=0))
                print("class 1234 ground truth:", np.sum(g, axis=0))
                
                print(classification_report(g, p, target_names=["problem", "treatment", "test"]))


if __name__ == "__main__":
    dataset_test()
