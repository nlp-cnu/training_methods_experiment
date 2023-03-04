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



# The point of this experiment is to see if bert learns meta information about downstream tasks during intermediate fine-tuning on general English corpus.
# There is no potential unlearning of domain-specific information because bert-base is trained only on general English

def run_experiment_2_meta():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    logging.set_verbosity("ERROR")
    # If there is an existing results file, get rid of it
    final_results_file = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_2_META_RESULTS, FINAL_RESULTS_FILE)
    if os.path.isfile(final_results_file):
        os.remove(final_results_file)
    # Set the header of the results file, getting macro & micro precision, recall, and f1s
    with open(final_results_file, "a+") as f:
        f.write("dataset\tlm_name\tmicro_precision_av\tmicro_precision_std\tmicro_recall_av\tmicro_recall_std\tmicro_f1_av\tmicro_f1_std\tmacro_precision_av\tmacro_precision_std\tmacro_recall_av\tmacro_recall_std\tmacro_f1_av\tmacro_f1_std\n")

    language_model = 'bert-base-uncased'

    persistent_language_model = language_model  # Tracking to get right tokenizer
    onto_class_map = DATASET_TO_CLASS_MAP[ONTO_DATA.split(os.sep)[-1]]
    onto_num_classes = len(onto_class_map)

    onto_file_path = os.path.join(ONTO_DATA, CONVERTED_DATASET_FILE)
    onto_data = Token_Classification_Dataset(onto_file_path, onto_num_classes, language_model, seed=SEED)
    onto_train_data = onto_data.data
    onto_train_labels = onto_data.labels
    onto_train_data, onto_val_data, onto_train_labels, onto_val_labels = train_test_split(onto_train_data, onto_train_labels, test_size=VALIDATION_SIZE, random_state=SEED)

    onto_classifier = MultiClass_Token_Classifier(language_model, onto_num_classes)
    onto_val_csv_log_file = os.path.join(test_results_path, f"ONTO_{language_model_name}_validation.csv")
    onto_classifier.train(onto_train_data, onto_train_labels, validation_data=(onto_val_data, onto_val_labels), csv_log_file=onto_val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE)

    # Saving the mode
    onto_lm_loc = os.path.join("..", "models", f"{language_model_name}_ONTO")
    onto_classifier.save_language_model(onto_lm_loc)

    for dataset_path zip(DOMAIN_SPECIFIC_DATASETS):
        if language_model == "NONE":
            continue
        dataset_name = dataset_path.split(os.sep)[-1]
        print("Dataset:", dataset_name)
        class_map = DATASET_TO_CLASS_MAP[dataset_name]
        num_classes = len(class_map)
        print("Class mapping:", class_map)

        language_model_name = language_model.split(os.sep)[-1]
        print("\tLanguage model:" + language_model_name)
        
        training_file_path = os.path.join(dataset_path, CONVERTED_DATASET_FILE)
        test_results_path = os.path.join(RESULTS_DIR_PATH, EXPERIMENT_2_META_RESULTS)
        Path(test_results_path).mkdir(parents=True, exist_ok=True)

        data = Token_Classification_Dataset(training_file_path, num_classes, language_model, seed=SEED)
        folds = list(data.get_folds(NUM_FOLDS))

        predictions = []
        golds = []

        # Train on onto data and save language model before CV


        for index, train_test in enumerate(folds):
            train_index, test_index = train_test
            train_data = np.array(data.data)[train_index]
            train_labels = np.array(data.labels)[train_index]
            test_data = np.array(data.data)[test_index]
            test_labels = np.array(data.labels)[test_index]

            train_data_, val_data, train_labels_, val_labels = train_test_split(train_data, train_labels, test_size=VALIDATION_SIZE, random_state=SEED)

            classifier = MultiClass_Token_Classifier(onto_lm_loc, num_classes, tokenizer=persistent_language_model)
            val_csv_log_file = os.path.join(test_results_path, f"{dataset_name}_{language_model_name}_validation_{index}.csv")
            validation_metrics = classifier.train(train_data_, train_labels_, validation_data=(val_data, val_labels), csv_log_file=val_csv_log_file, early_stop_patience=EARLY_STOPPING_PATIENCE)
            validation_history = validation_metrics.history
            target_metric = validation_history['val_micro_f1']
            
            num_epochs = target_metric.index(max(target_metric))
            
            classifier = MultiClass_Token_Classifier(onto_lm_loc, num_classes, tokenizer=persistent_language_model)
            test_csv_log_file = os.path.join(test_results_path, f"{dataset_name}_{language_model_name}_test_{index}.csv")
            classifier.train(train_data, train_labels, epochs=num_epochs, csv_log_file=test_csv_log_file)


            predictions.append(classifier.predict(test_data))
            golds.append(test_labels)
            
            K.clear_session()
            gc.collect()
            del classifier

        pred_micro_precisions = []
        pred_macro_precisions = []
        pred_micro_recalls = []
        pred_macro_recalls = []
        pred_micro_f1s = []
        pred_macro_f1s = []

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
            f.write(f"{dataset_name}\t{language_model_name}\t{micro_precision_av}\t{micro_precision_std}\t{micro_recall_av}\t{micro_recall_std}\t{micro_f1_av}\t{micro_f1_std}\t{macro_precision_av}\t{macro_precision_std}\t{macro_recall_av}\t{macro_recall_std}\t{macro_f1_av}\t{macro_f1_std}\n")

            

if __name__ == "__main__":
    run_experiment_2_meta()


