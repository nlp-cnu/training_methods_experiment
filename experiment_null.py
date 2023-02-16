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
    for dataset_path in DOMAIN_SPECIFIC_DATASETS[:1]:
        dataset_name = dataset_path.split(os.sep)[-1]
        print("Dataset:", dataset_name)
        class_map = DATASET_TO_CLASS_MAP[dataset_name]
        num_classes = len(class_map)
        print("Class mapping:", class_map)
        for language_model in ALL_MODELS[0:1]:
            language_model = os.path.join("..", "models", "new_model")
            classifier = MultiClass_Token_Classifier(language_model, num_classes)

            language_model_name = language_model.split(os.sep)[-1]
            
            print("\tLanguage model:" + language_model_name)

            training_file_path = os.path.join(dataset_path, CONVERTED_DATASET_FILE_MINI)
            data = Token_Classification_Dataset(training_file_path, num_classes, language_model, seed=SEED)
            folds = list(data.get_folds(2))

            test_micro_f1 = []
            test_macro_f1 = []


            predictions = []
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
                # print(f"Validation history fold_{index}:", validation_history)
                target_metric = validation_history['val_micro_f1']
                
                num_epochs = target_metric.index(max(target_metric)) + 1
                
                classifier = MultiClass_Token_Classifier(language_model, num_classes)
                # print(f"Test history fold_{index}:", classifier.train(train_data, train_labels, epochs=num_epochs).history)

                classifier.save_weights("../models/new_model")

                ps = classifier.predict(test_data)
                predictions.append(classifier.predict(test_data))
                golds.append(test_labels)

                # ^^^^^^^^^^^^^^

            pred_micro_precisions = []
            pred_macro_precisions = []
            pred_micro_recalls = []
            pred_macro_recalls = []
            pred_micro_f1s = []
            pred_macro_f1s = []

            print()
            print("Model test scores using predict:")

            # For each fold there is a y_true and y_pred
            for p, g in zip(predictions, golds):
                # making y_pred and y_true have the same size by trimming
                num_samples = p.shape[0]
                max_num_tokens_in_batch = p.shape[1]
                # Transforms g to the same size as P
                # removes the NONE class
                g = g[:, :max_num_tokens_in_batch, :]

                gt_final = []
                pred_final = []

                for sample_pred, sample_gt, i in zip(p, g, range(num_samples)):
                    # vv Find where the gt labels stop (preds will be junk after this) and trim the labels and predictions vv
                    trim_index = 0
                    while trim_index < len(sample_gt) and not all(v == 0 for v in sample_gt[trim_index]):
                        trim_index += 1
                    sample_gt = sample_gt[:trim_index, :]
                    for s in sample_gt:
                        gt_final.append(s.tolist())

                    sample_pred = (sample_pred == sample_pred.max(axis=1)[:,None]).astype(int)
                    sample_pred = sample_pred[:trim_index, :]
                    for s in sample_pred:
                        pred_final.append(s.tolist())

                    # ^^^^^
                # Transforming the predictions and labels so that the NONE class is not counted
                p = np.array(pred_final)
                g = np.array(gt_final)

                p = p.reshape((-1, num_classes))[:, 1:]
                g = g.reshape((-1, num_classes))[:, 1:]

                # Calculating the metrics w/ sklearn
                binary_task = len(class_map) == 2
                print("Is this a binary task:", binary_task)

                if binary_task:
                    target_names = list(class_map)
                    report_metrics = classification_report(g, p, target_names=target_names, digits=3, output_dict=True)

                    # collecting the reported metrics
                    # The macro and micro f1 scores are the same for the binary classification task
                    micro_averaged_stats = report_metrics["macro avg"]
                    micro_precision = micro_averaged_stats["precision"]
                    pred_micro_precisions.append(micro_precision)
                    micro_recall = micro_averaged_stats["recall"]
                    pred_micro_recalls.append(micro_recall)
                    micro_f1 = micro_averaged_stats["f1-score"]
                    pred_micro_f1s.append(micro_f1)

                    macro_averaged_stats = report_metrics["macro avg"]
                    macro_precision = macro_averaged_stats["precision"]
                    pred_macro_precisions.append(macro_precision)
                    macro_recall = macro_averaged_stats["recall"]
                    pred_macro_recalls.append(macro_recall)
                    macro_f1 = macro_averaged_stats["f1-score"]
                    pred_macro_f1s.append(macro_f1)


                else:
                    target_names = list(class_map)[1:]
                    report_metrics = classification_report(g, p, target_names=target_names, digits=3, output_dict=True)

                    # collecting the reported metrics
                    micro_averaged_stats = report_metrics["micro avg"]
                    micro_precision = micro_averaged_stats["precision"]
                    pred_micro_precisions.append(micro_precision)
                    micro_recall = micro_averaged_stats["recall"]
                    pred_micro_recalls.append(micro_recall)
                    micro_f1 = micro_averaged_stats["f1-score"]
                    pred_micro_f1s.append(micro_f1)

                    macro_averaged_stats = report_metrics["macro avg"]
                    macro_precision = macro_averaged_stats["precision"]
                    pred_macro_precisions.append(macro_precision)
                    macro_recall = macro_averaged_stats["recall"]
                    pred_macro_recalls.append(macro_recall)
                    macro_f1 = macro_averaged_stats["f1-score"]
                    pred_macro_f1s.append(macro_f1)

                print(report_metrics)

            # Writing the reported metrics to file
            micro_precision_av = np.mean(pred_micro_precisions)
            micro_precision_std = np.std(pred_micro_precisions)
            micro_recall_av = np.mean(pred_micro_recalls)
            micro_recall_std = np.std(pred_micro_recalls)
            micro_f1_av = np.mean(pred_micro_f1s)
            micro_f1_std = np.std(pred_micro_f1s)

            macro_precision_av = np.mean(pred_macro_precisions)
            macro_precision_std = np.std(pred_macro_precisions)
            macro_recall_av = np.mean(pred_macro_recalls)
            macro_recall_std = np.std(pred_macro_recalls)
            macro_f1_av = np.mean(pred_macro_f1s)
            macro_f1_std = np.std(pred_macro_f1s)


            print("Micro averaged stats:")
            print(f"p:av={np.mean(pred_micro_precisions)}, std={np.std(pred_micro_precisions)}")
            print(f"r:av={np.mean(pred_micro_recalls)}, std={np.std(pred_micro_recalls)}")
            print(f"f1:av={np.mean(pred_micro_f1s)}, std={np.std(pred_micro_f1s)}")
            print("Macro averaged stats:")
            print(f"p:av={np.mean(pred_macro_precisions)}, std={np.std(pred_macro_precisions)}")
            print(f"r:av={np.mean(pred_macro_recalls)}, std={np.std(pred_macro_recalls)}")
            print(f"f1:av={np.mean(pred_macro_f1s)}, std={np.std(pred_macro_f1s)}")


if __name__ == "__main__":
    dataset_test()
