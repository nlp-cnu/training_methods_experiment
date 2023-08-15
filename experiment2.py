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
from utilities.Evaluator import *

def run_experiment_2():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    logging.set_verbosity("ERROR")
    # If there is an existing results file, get rid of it
    final_results_file = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_2_RESULTS, FINAL_RESULTS_FILE)
    if os.path.isfile(final_results_file):
        os.remove(final_results_file)
    # Set the header of the results file, getting macro & micro precision, recall, and f1s
    with open(final_results_file, "a+") as f:
        f.write("dataset\tlm_name\tmicro_precision_av\tmicro_precision_std\tmicro_recall_av\tmicro_recall_std\tmicro_f1_av\tmicro_f1_std\tmacro_precision_av\tmacro_precision_std\tmacro_recall_av\tmacro_recall_std\tmacro_f1_av\tmacro_f1_std\t")

        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " micro precision\t")
        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " micro recall\t")
        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " micro f1\t")
        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " macro precision\t")
        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " macro recall\t")
        for i in range(1, NUM_FOLDS + 1):
            f.write("fold " + str(i) + " macro f1\t")
        f.write("\n")


    # Iterate over each dataset
    for dataset_path, language_model in zip(DOMAIN_SPECIFIC_DATASETS, EXP1_WINNING_MODELS):
        if language_model == "NONE":
            continue
        dataset_name = dataset_path.split(os.sep)[-1]
        print("Dataset:", dataset_name)
        class_map = DATASET_TO_CLASS_MAP[dataset_name]
        num_classes = len(class_map)
        print("Class mapping:", class_map)

        language_model_name = language_model.split(os.sep)[-1]
        print("\tLanguage model:" + language_model_name)

        # set up output paths
        training_file_path = os.path.join(dataset_path, CONVERTED_DATASET_FILE)
        test_results_path = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_1_RESULTS)
        Path(test_results_path).mkdir(parents=True, exist_ok=True)

        # create the tokenizer - it must be consistent across classifier and dataset
        tokenizer = AutoTokenizer.from_pretrained(language_model)

        # set the max_num_tokens
        # bertweet has a max_num_tokens of 128, all others are MAX_NUM_TOKENS=512
        if 'bertweet' in language_model:
            max_num_tokens = 128
        else:
            max_num_tokens = MAX_NUM_TOKENS

        ## 1. TRAIN THE ENCODER ON ONTO DATA
        persistent_language_model = language_model  # Tracking to get right tokenizer
        onto_class_map = DATASET_TO_CLASS_MAP[ONTO_DATA.split(os.sep)[-1]]
        onto_num_classes = len(onto_class_map)

        # load the onto data
        onto_file_path = os.path.join(ONTO_DATA, CONVERTED_DATASET_FILE)
        onto_data = Token_Classification_Dataset(onto_file_path, onto_num_classes, tokenizer, seed=SEED,
                                                 max_num_tokens=max_num_tokens)
        onto_train_data = onto_data.data
        onto_train_labels = onto_data.labels
        onto_train_data, onto_val_data, onto_train_labels, onto_val_labels = train_test_split(onto_train_data, onto_train_labels,
                                                                                              test_size=VALIDATION_SIZE,
                                                                                              random_state=SEED, shuffle=True)
        # Crate and train the classifier on the onto data
        onto_classifier = MultiClass_Token_Classifier(language_model, onto_num_classes, tokenizer, max_num_tokens)
        if PARTIAL_UNFREEZING:
            print("Training the onto-Decoder only")
            # train the decoder
            onto_classifier.language_model.trainable = False
            onto_val_csv_log_file = os.path.join(test_results_path, f"ONTO_{language_model_name}_validation_decoder.csv")
            onto_classifier.train(onto_train_data, onto_train_labels, validation_data=(onto_val_data, onto_val_labels),
                                  csv_log_file=onto_val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE,
                                  restore_best_weights=True)
        # train the whole network
        onto_classifier.language_model.trainable = True
        onto_val_csv_log_file = os.path.join(test_results_path, f"ONTO_{language_model_name}_validation.csv")
        onto_classifier.train(onto_train_data, onto_train_labels, validation_data=(onto_val_data, onto_val_labels),
                              csv_log_file=onto_val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True)

        # Save the decoder (language model) weights
        onto_lm_loc = os.path.join("..", "models", f"{language_model_name}_ONTO")
        onto_classifier.save_language_model(onto_lm_loc)


        ## 2. PERFORM CROSS-VALIDATION
        # load the training data and split into folds
        data = Token_Classification_Dataset(training_file_path, num_classes, tokenizer, seed=SEED,
                                            max_num_tokens=max_num_tokens)
        folds = list(data.get_folds(NUM_FOLDS))

        # loop through each fold and collect results
        predictions = []
        golds = []
        for index, train_test in enumerate(folds):
            # get train, validation, test data for this fold
            train_index, test_index = train_test
            train_data = []
            train_labels = []
            for sample_index in train_index:
                train_data.append(data.data[sample_index])
                train_labels.append(data.labels[sample_index])
            test_data = []
            test_labels = []
            for sample_index in test_index:
                test_data.append(data.data[sample_index])
                test_labels.append(data.labels[sample_index])

            # get the validation split
            train_data_, val_data, train_labels_, val_labels = train_test_split(train_data, train_labels,
                                                                                test_size=VALIDATION_SIZE,
                                                                                random_state=SEED, shuffle=True)

            # create and train the classifier with or without partial unfreezing
            classifier = MultiClass_Token_Classifier(onto_lm_loc, num_classes, tokenizer, max_num_tokens)
            if PARTIAL_UNFREEZING:
                print("Training the decoder only")
                # train the decoder
                val_csv_log_file = os.path.join(test_results_path,
                                                f"{dataset_name}_{language_model_name}_validation_decoder_{index}.csv")
                classifier.language_model.trainable = False
                classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels),
                                 csv_log_file=val_csv_log_file,
                                 early_stop_patience=EARLY_STOPPING_PATIENCE,
                                 restore_best_weights=True)

            # train the whole network
            classifier.language_model.trainable = True
            val_csv_log_file = os.path.join(test_results_path,
                                            f"{dataset_name}_{language_model_name}_validation_{index}.csv")
            classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels),
                             csv_log_file=val_csv_log_file,
                             early_stop_patience=EARLY_STOPPING_PATIENCE,
                             restore_best_weights=True)

            # get the test set predictions
            predictions.append(classifier.predict(test_data))
            golds.append(test_labels)

            # I think there are some memory leaks within keras, so do some garbage collecting
            K.clear_session()
            gc.collect()
            del classifier

        ## 2. OUTPUT CROSS-VALIDATION RESULTS FOR THIS DATASET
        collect_and_output_results(predictions, golds, class_map, final_results_file, dataset_name,
                                   language_model_name)

if __name__ == "__main__":
    run_experiment_2()


