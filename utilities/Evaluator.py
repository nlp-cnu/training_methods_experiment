import numpy as np

def evaluate_predictions(pred_y, true_y, class_names):
    """
    Evaluates the predictions against true values. Predictions and Gold are from the classifier/dataset.
    They are a 3-D matrix [line, token, one-hot-vector of class]

    :param pred_y: matrix of predicted labels (one-hot encoded 0's and 1's)
    :param true_y: matrix of true values
    :param class_names: an ordered list of class names (strings)
    :param report_none: if True, then results for the none class will be reported and averaged into micro
              and macro scores. A None class is automatically added to the class_names
    :param multi_class: indicates if the labels are multi-class or multi-label
    """

    binary_classification = False
    if len(class_names) == 1:
        binary_classification = True
        class_names = ['none'] + class_names

    # grab dimensions
    num_lines = pred_y.shape[0]
    padded_token_length = pred_y.shape[1]
    num_classes = pred_y.shape[2]

    # ensure the num predicted lines = num gold lines
    if num_lines != len(true_y):
        print("ERROR: the number of predicted lines does not equal the number of gold lines. "
              f"\n  num predicted lines = {num_lines}, num gold_lines = {len(true_y)}"
              "\n   Do your prediction and gold datasets match?")
        exit()

    # ensure the class_names length matches the number of predicted classes
    if num_classes != len(class_names):
        print("ERROR in evaluate_predictions: number of predicted classes not equal to the number of "
              + "provided class names. Did you forget about a None class?")
        exit()

    # flatten the predictions. So, it is one prediction per token
    gold_flat = []
    pred_flat = []
    for i in range(num_lines):
        # get the gold and predictions for this line
        line_gold = true_y[i][:, :]
        line_pred = pred_y[i, :, :]

        # the gold contains the number of tokens (predictions are padded)
        # remove padded predictions
        num_tokens = line_gold.shape[0]
        line_pred = pred_y[i, :num_tokens, :]

        # convert token classifications to categorical.
        if binary_classification:  # multilabel or binary
            # Argmax returns 0 if everything is 0,
            # so, determine if classification is None class. If it's not, add 1 to the argmax
            not_none = np.max(line_gold, axis=1) > 0
            line_gold_categorical = np.argmax(line_gold, axis=1) + not_none
            not_none = np.max(line_pred, axis=1) > 0
            line_pred_categorical = np.argmax(line_pred, axis=1) + not_none
        else:
            line_gold_categorical = np.argmax(line_gold, axis=1)
            line_pred_categorical = np.argmax(line_pred, axis=1)

        # add to the flattened list of labels
        gold_flat.extend(line_gold_categorical.tolist())
        pred_flat.extend(line_pred_categorical.tolist())

    # initialize the dictionaries
    num_classes = len(class_names)
    tp = []
    fp = []
    fn = []
    for i in range(num_classes):
        tp.append(0)
        fp.append(0)
        fn.append(0)

    # count the tps, fps, fns
    num_samples = len(pred_flat)
    for i in range(num_samples):

        # calculating tp, fp, fn for multiclass and binary is slightly different
        if not binary_classification:
            true_index = gold_flat[i]
            pred_index = pred_flat[i]
            correct = pred_flat[i] == gold_flat[i]
            if correct:
                tp[true_index] += 1
            else:
                fp[pred_index] += 1
                fn[true_index] += 1

        if binary_classification:
            if gold_flat[i] == 1 and pred_flat[i] == 1:
                tp[0] += 1
            elif gold_flat[i] == 0 and pred_flat[i] == 1:
                fp[0] += 1
            elif gold_flat[i] == 1 and pred_flat[i] == 0:
                fn[0] += 1
            # elif gold_flat[i] == 0 and pred_flat[i] == 0:
            # tn += 1

    # convert tp, fp, fn into arrays and trim if not reporting none
    if binary_classification:  # report for all classes (binary)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
    else:  # report for all but the None class (multiclass typically)
        # take [1:] to remove the None Class
        tp = np.array(tp)[1:]
        fp = np.array(fp)[1:]
        fn = np.array(fn)[1:]

    # remove 'None' from class_names
    class_names = class_names[1:]

    # account for 0s which will result in division by 0
    if tp[tp == 0] += 1e-10
    
    # calculate precision, recall, and f1 for each class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    support = tp + fn

    # calculate micro and macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    all_tp = np.sum(tp)
    all_fp = np.sum(fp)
    all_fn = np.sum(fn)
    micro_precision = all_tp / (all_tp + all_fp)
    micro_recall = all_tp / (all_tp + all_fn)

    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    micro_averaged_stats = {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}
    macro_avareged_stats = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1}

    return micro_averaged_stats, macro_avareged_stats


def collect_and_output_results(predictions, golds, class_map, final_results_file, dataset_name, language_model_name):
    pred_micro_precisions = []
    pred_macro_precisions = []
    pred_micro_recalls = []
    pred_macro_recalls = []
    pred_micro_f1s = []
    pred_macro_f1s = []

    # TODO _ DELETE THIS PICKLE PORTION --- BUT USEFUL FOR DEBUGGING
    import pickle
    with open('temp_pred_file.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    with open('temp_gold_file.pkl', 'wb') as file:
        pickle.dump(golds, file)
    #with open('temp_pred_file.pkl', 'rb') as file:
    #    predictions = pickle.load(file)
    #with open('temp_gold_file.pkl', 'rb') as file:
    #   golds = pickle.load(file)

    # For each fold there is a y_true and y_pred
    for p, g in zip(predictions, golds):

        # get the class names
        class_names = list(class_map)
        binary_task = len(class_map) == 2
        if binary_task:
            class_names = list(class_map)[1:]

        # get statistics
        micro_averaged_stats, macro_averaged_stats = evaluate_predictions(p, g, class_names)

        # record statistics
        micro_precision = micro_averaged_stats["precision"]
        pred_micro_precisions.append(micro_precision)
        micro_recall = micro_averaged_stats["recall"]
        pred_micro_recalls.append(micro_recall)
        micro_f1 = micro_averaged_stats["f1"]
        pred_micro_f1s.append(micro_f1)

        macro_precision = macro_averaged_stats["precision"]
        pred_macro_precisions.append(macro_precision)
        macro_recall = macro_averaged_stats["recall"]
        pred_macro_recalls.append(macro_recall)
        macro_f1 = macro_averaged_stats["f1"]
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
