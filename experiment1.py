from argparse import ArgumentParser
import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import logging

from utilities.Classifier import *
from utilities.Dataset import *
from utilities.constants import *


def run_experiment_1():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    logging.set_verbosity("ERROR")
    final_results_file = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_1_RESULTS, FINAL_RESULTS_FILE)
    if os.path.isfile(final_results_file):
        os.remove(final_results_file)
    with open(final_results_file, "a+") as f:
        f.write("dataset\tlm_name\tmicro_f1_av\tmicro_f1_std\tmacro_f1_av\tmacro_f1_std\n")
    for dataset_path in DOMAIN_SPECIFIC_DATASETS:
        dataset_name = dataset_path.split(os.sep)[-1]
        print("Dataset:", dataset_name)
        class_map = DATASET_TO_CLASS_MAP[dataset_name]
        num_classes = len(class_map)
        print("Class mapping:", class_map)
        for language_model in ALL_MODELS:
            language_model_name = language_model.split(os.sep)[-1]
            print("\tLanguage model:" + language_model_name)
            
            training_file_path = os.path.join(dataset_path, CONVERTED_DATASET_FILE)
            
            test_results_path = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_1_RESULTS)
            Path(test_results_path).mkdir(parents=True, exist_ok=True)

            data = Token_Classification_Dataset(training_file_path, num_classes, language_model, seed=SEED)
            folds = list(data.get_folds(NUM_FOLDS))

            test_micro_f1 = []
            test_macro_f1 = []

            for index, train_test in enumerate(folds):
                train_index, test_index = train_test
                train_data = np.array(data.data)[train_index]
                train_labels = np.array(data.labels)[train_index]
                test_data = np.array(data.data)[test_index]
                test_labels = np.array(data.labels)[test_index]

                train_data_, val_data, train_labels_, val_labels = train_test_split(train_data, train_labels, test_size=VALIDATION_SIZE, random_state=3)

                classifier = MultiClass_Token_Classifier(language_model, num_classes)
                val_csv_log_file = os.path.join(test_results_path, f"{dataset_name}_{language_model_name}_validation_{index}.csv")
                validation_metrics = classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels), csv_log_file=val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE)
                validation_history = validation_metrics.history
                target_metric = validation_history['val_micro_f1']
                
                num_epochs = target_metric.index(max(target_metric))
                
                classifier = MultiClass_Token_Classifier(language_model, num_classes)
                test_csv_log_file = os.path.join(test_results_path, f"{dataset_name}_{language_model_name}_test_{index}.csv")
                classifier.train(train_data, train_labels, epochs=num_epochs, csv_log_file=test_csv_log_file)

                predictions = classifier.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
                test_micro_f1.append(predictions[1])
                test_macro_f1.append(predictions[2])
                
                K.clear_session()
                gc.collect()
                del classifier

            micro_f1_mean = np.mean(test_micro_f1)
            micro_f1_stddev = np.std(test_micro_f1)
            macro_f1_mean = np.mean(test_macro_f1)
            macro_f1_stddev = np.std(test_macro_f1)

            with open(final_results_file, "a+") as f:
                f.write(f"{dataset_name}\t{language_model_name}\t{micro_f1_mean}\t{micro_f1_stddev}\t{macro_f1_mean}\t{macro_f1_stddev}\n")

            

if __name__ == "__main__":
    run_experiment_1()


