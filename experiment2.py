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
        f.write("dataset\tlm_name\tmicro_precision_av\tmicro_precision_std\tmicro_recall_av\tmicro_recall_std\tmicro_f1_av\tmicro_f1_std\tmacro_precision_av\tmacro_precision_std\tmacro_recall_av\tmacro_recall_std\tmacro_f1_av\tmacro_f1_std\n")

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
    for dataset_path, language_model in zip(DOMAIN_SPECIFIC_DATASETS[4:5], EXP1_WINNING_MODELS[4:5]):
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
        tokenizer_name = language_model_name
        if 'bertweet' in language_model:
            tokenizer_name = 'vinai/bertweet-base'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        # set the max_num_tokens
        # bertweet has a max_num_tokens of 128, all others are MAX_NUM_TOKENS=512
        if 'bertweet' in language_model:
            max_num_tokens = 128
        else:
            max_num_tokens = MAX_NUM_TOKENS


        ## FIRST - TRAIN ON ONTO DATA AND SAVE LANGUAGE MODEL
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
                                  restore_best_weights=True, epochs=2) #TODO - restore to more than 1 epoch)
        # train the whole network
        onto_classifier.language_model.trainable = True
        onto_val_csv_log_file = os.path.join(test_results_path, f"ONTO_{language_model_name}_validation.csv")
        onto_classifier.train(onto_train_data, onto_train_labels, validation_data=(onto_val_data, onto_val_labels),
                              csv_log_file=onto_val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True, epochs=2) #TODO - restore to more than 1 epoch)

        # Save the decoder (language model) weights
        onto_lm_loc = os.path.join("..", "models", f"{language_model_name}_ONTO")
        onto_classifier.save_language_model(onto_lm_loc)


        ## SECOND - PERFORM CROSS-VALIDATION
        # load the training data and split into folds
        data = Token_Classification_Dataset(training_file_path, num_classes, tokenizer, seed=SEED,
                                            max_num_tokens=max_num_tokens)
        folds = list(data.get_folds(NUM_FOLDS))

        # loop through each fold and collect results
        predictions = []
        golds = []
        for index, train_test in enumerate(folds):
            # split the data into train, validation, and test sets
            train_index, test_index = train_test
            train_data = np.array(data.data)[train_index]
            train_labels = np.array(data.labels)[train_index]
            test_data = np.array(data.data)[test_index]
            test_labels = np.array(data.labels)[test_index]

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
                                 restore_best_weights=True, epochs=2) #TODO - restore to more than 1 epoch))

            # train the whole network
            classifier.language_model.trainable = True
            val_csv_log_file = os.path.join(test_results_path,
                                            f"{dataset_name}_{language_model_name}_validation_{index}.csv")
            classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels),
                             csv_log_file=val_csv_log_file,
                             early_stop_patience=EARLY_STOPPING_PATIENCE,
                             restore_best_weights=True, epochs=2) #TODO - restore to more than 1 epoch))

            # get the test set predictions
            predictions.append(classifier.predict(test_data))
            golds.append(test_labels)

            # I think there are some memory leaks within keras, so do some garbage collecting
            K.clear_session()
            gc.collect()
            del classifier

        ## collect statistics from cross-validation
        pred_micro_precisions = []
        pred_macro_precisions = []
        pred_micro_recalls = []
        pred_macro_recalls = []
        pred_micro_f1s = []
        pred_macro_f1s = []

        # For each fold there is a y_true and y_pred
        for p, g in zip(predictions, golds):

            # get the class names
            class_names = list(class_map)
            binary_task = len(class_map) == 2
            if binary_task:
                class_names = list(class_map[1:])

            # get statistics
            micro_averaged_stats, macro_averaged_stats = evaluate_predictions(p, g, class_names)

            # record statistics
            micro_precision = micro_averaged_stats["precision"]
            pred_micro_precisions.append(micro_precision)
            micro_recall = micro_averaged_stats["recall"]
            pred_micro_recalls.append(micro_recall)
            micro_f1 = micro_averaged_stats["f1-score"]
            pred_micro_f1s.append(micro_f1)

            macro_precision = macro_averaged_stats["precision"]
            pred_macro_precisions.append(macro_precision)
            macro_recall = macro_averaged_stats["recall"]
            pred_macro_recalls.append(macro_recall)
            macro_f1 = macro_averaged_stats["f1-score"]
            pred_macro_f1s.append(macro_f1)

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

        with open(final_results_file, "a+") as f:
            # write results for averaged performance
            f.write(
                f"{dataset_name}\t{language_model_name}\t{micro_precision_av}\t{micro_precision_std}\t{micro_recall_av}\t{micro_recall_std}\t{micro_f1_av}\t{micro_f1_std}\t{macro_precision_av}\t{macro_precision_std}\t{macro_recall_av}\t{macro_recall_std}\t{macro_f1_av}\t{macro_f1_std}\t")

            # and write the stats per fold (so statistical significance can be computed
            f.write('\t'.join(str(num) for num in pred_micro_precisions) + "\t")
            f.write('\t'.join(str(num) for num in pred_micro_recalls) + "\t")
            f.write('\t'.join(str(num) for num in pred_micro_f1s) + "\t")

            f.write('\t'.join(str(num) for num in pred_macro_precisions) + "\t")
            f.write('\t'.join(str(num) for num in pred_macro_recalls) + "\t")
            f.write('\t'.join(str(num) for num in pred_macro_f1s))

            f.write("\n")

if __name__ == "__main__":
    run_experiment_2()


