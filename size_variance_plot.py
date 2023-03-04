import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit


def linear_func(x, m, b):
    return m * x + b


def main():
    # THESE NEED TO BE ALPHABETIZED
    # Includes all datasets
    # dataset_word_size = [291345, 922104, 1481246, 271175, 463200, 364446, 904414, 146791, 729228]
    # dataset_sample_size = [18300, 42575, 127125, 13880, 19988, 43940, 12809, 7153, 33842]
    
    # Excludes BC7MED, class imbalanced twitter dataset
    # dataset_word_size = np.array([291345, 922104, 271175, 463200, 364446, 904414, 146791, 729228])
    # dataset_sample_size = np.array([18300, 42575, 13880, 19988, 43940, 12809, 7153, 33842])

    # Only looks at the number of positive samples 
    sample_size_pos = np.array([1300, 35020, 311, 11542, 19988, 22489, 7016, 3937, 16728])
    # Excludes Bc7
    # sample_size_pos = np.array([1300, 35020, 11542, 19988, 22489, 7016, 3937, 16728])

    file_path = os.path.join("results", "experiment_1_results", "final_results_master.tsv")
    dtypes = {'dataset':str, 'lm_name':str, "micro_precision_av":float, "micro_precision_std":float, "micro_recall_av":float, "micro_recall_std":float, "micro_f1_av":float, "micro_f1_std":float, "macro_precision_av":float, "macro_precision_std":float, "macro_recall_av":float, "macro_recall_std":float, "macro_f1_av":float, "macro_f1_std":float}
    df = pd.read_csv(file_path, header=0, sep='\t', dtype=dtypes)

    # Dropping bc7med because its a major outlier
    # df.drop(df[df.dataset=='bc7med'].index, inplace=True)
    var_by_dataset = df.groupby(['dataset']).var(numeric_only=True)
    # THESE ARE ALPHABETIZED
    variances = var_by_dataset['micro_f1_av'].tolist()

    # Linear regression for the word count by variance
#     popt, pcov = curve_fit(linear_func, dataset_word_size, variances)
#     word_residuals = variances - linear_func(dataset_word_size, popt[0], popt[1])
#     word_ss_res = np.sum(word_residuals ** 2)
#     word_ss_tot = np.sum((variances - np.mean(variances))**2)
# 
#     word_r_squared = 1 - (word_ss_res / word_ss_tot)
# 
#     print("OPT params:", popt)
#     print("COVs:", pcov)
#     print("R^2:", word_r_squared)
#     print("-" * 50)
# 
#     fig, ax = plt.subplots(figsize=[12,8])
# 
#     ax.set_title("Dataset word count by variance", fontsize=20)
#     ax.set_xlabel("Number of words in dataset", fontsize=15)
#     ax.set_title("Variance in micro f1 scores", fontsize=15)
#     ax.scatter(dataset_word_size, variances, linewidth=5)
#     ax.plot(dataset_word_size, [linear_func(val, popt[0], popt[1]) for val in dataset_word_size], label=r"$r^{2}=$" + f"{word_r_squared:.4f}") 
# 
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, loc='upper right', fontsize=15)
# 
#     plt.tight_layout()
#     plt.savefig("results/dataset/word_count_variance.pdf", dpi=400)


    # Linear regression for the sample count by variance
    popt, pcov = curve_fit(linear_func, sample_size_pos, variances)
    sample_residuals = variances - linear_func(sample_size_pos, popt[0], popt[1])
    sample_ss_res = np.sum(sample_residuals ** 2)
    sample_ss_tot = np.sum((variances - np.mean(variances))**2)

    sample_r_squared = 1 - (sample_ss_res / sample_ss_tot)

    print("OPT params:", popt)
    print("COVs:", pcov)
    print("R^2:", sample_r_squared)

    fig, ax = plt.subplots(figsize=[12,8])

    ax.set_title("Dataset sample count by variance", fontsize=20)
    ax.set_xlabel("Number of positive samples in dataset", fontsize=15)
    ax.set_title("Variance in micro f1 scores", fontsize=15)
    # ax.set_ylim([0, 0.007])
    ax.scatter(sample_size_pos, variances, linewidth=5)
    ax.plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in sample_size_pos], label=r"$r^{2}=$" + f"{sample_r_squared:.2f}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig("results/dataset/pos_sample_count_variance.pdf", dpi=400)


    # Doing this to check that the means increase with more positive samples
    df = pd.read_csv(file_path, header=0, sep='\t', dtype=dtypes)
    # Dropping bc7med because its a major outlier
    # df.drop(df[df.dataset=='bc7med'].index, inplace=True)
    mean_by_dataset = df.groupby(['dataset']).mean(numeric_only=True)
    # THESE ARE ALPHABETIZED
    means = mean_by_dataset['micro_f1_av'].tolist()

    popt, pcov = curve_fit(linear_func, sample_size_pos, means)
    sample_residuals = means - linear_func(sample_size_pos, popt[0], popt[1])
    sample_ss_res = np.sum(sample_residuals ** 2)
    sample_ss_tot = np.sum((means - np.mean(means))**2)

    mean_r_squared = 1 - (sample_ss_res / sample_ss_tot)

    print("OPT params:", popt)
    print("COVs:", pcov)
    print("R^2:", mean_r_squared)

    fig, ax = plt.subplots(figsize=[12,8])

    ax.set_title("Dataset sample count by mean", fontsize=20)
    ax.set_xlabel("Number of positive samples in dataset", fontsize=15)
    ax.set_title("Mean of micro f1 scores", fontsize=15)
    ax.scatter(sample_size_pos, means, linewidth=5)
    ax.plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in sample_size_pos], label=r"$r^{2}=$" + f"{mean_r_squared:.2f}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig("results/dataset/pos_sample_count_mean.pdf", dpi=400)



if __name__=="__main__":
    main()

