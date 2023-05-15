import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit


def linear_func(x, m, b):
    return m * x + b


def exp_func(x, a, b, c):
    return np.exp(b * x) + c


# def main():
#     # THESE NEED TO BE ALPHABETIZED
#     # ademiner, bc7dcpi, bc7med, cdr, cometa, i2c2, n2c2, ncbi, nlmchem
# 
#     # Includes all datasets
#     dataset_word_size = [291345, 922104, 1481246, 271175, 463200, 364446, 904414, 146791, 729228]
#     # dataset_sample_size = [18300, 42575, 127125, 13880, 19988, 43940, 12809, 7153, 33842]
#     
#     # Excludes BC7MED, class imbalanced twitter dataset
#     # dataset_word_size = np.array([291345, 922104, 271175, 463200, 364446, 904414, 146791, 729228])
#     # dataset_sample_size = np.array([18300, 42575, 13880, 19988, 43940, 12809, 7153, 33842])
# 
#     # Only looks at the number of positive samples 
#     sample_size_pos = np.array([1300, 35020, 311, 11542, 19988, 22489, 7016, 3937, 16728])
#     # Excludes Bc7
#     # sample_size_pos = np.array([1300, 35020, 11542, 19988, 22489, 7016, 3937, 16728])
# 
#     file_path = os.path.join("results", "experiment_1_results", "final_results_master.tsv")
#     dtypes = {'dataset':str, 'lm_name':str, "micro_precision_av":float, "micro_precision_std":float, "micro_recall_av":float, "micro_recall_std":float, "micro_f1_av":float, "micro_f1_std":float, "macro_precision_av":float, "macro_precision_std":float, "macro_recall_av":float, "macro_recall_std":float, "macro_f1_av":float, "macro_f1_std":float}
#     df = pd.read_csv(file_path, header=0, sep='\t', dtype=dtypes)
# 
#     var_by_dataset = df.groupby(['dataset']).var(numeric_only=True)
#     # THESE ARE ALPHABETIZED
#     variances = var_by_dataset['micro_f1_av'].tolist()
# 
#     # Linear regression for the sample count by variance
#     # sample_size_pos = np.log(sample_size_pos)
#     popt, pcov = curve_fit(linear_func, np.log(sample_size_pos), variances)
#     sample_residuals = variances - linear_func(np.log(sample_size_pos), popt[0], popt[1])
#     sample_ss_res = np.sum(sample_residuals ** 2)
#     sample_ss_tot = np.sum((variances - np.mean(variances))**2)
# 
#     sample_r_squared = 1 - (sample_ss_res / sample_ss_tot)
# 
#     print("OPT params:", popt)
#     print("COVs:", pcov)
#     print("R^2:", sample_r_squared)
# 
#     fig, ax = plt.subplots(1, 2, figsize=[12,8])
# 
#     ax[1].set_xlabel("Log of number of positive samples in dataset", fontsize=15)
#     ax[1].set_ylabel("F1 variance", fontsize=15)
#     # ax[1].set_title("Variance in micro f1 scores per positive sample size", fontsize=20)
#     # ax.set_ylim([0, 0.007])
#     ax[1].set_xscale('log')
#     ax[1].plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in np.log(sample_size_pos)], label=r"$r^{2}=$" + f"{sample_r_squared:.2f}", color="orangered", linewidth=5, zorder=0)
#     ax[1].scatter(sample_size_pos, variances, color='orange', linewidth=5, zorder=10)
#     # ax.plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in sample_size_pos])
# 
#     handles, labels = ax[1].get_legend_handles_labels()
#     ax[1].legend(handles, labels, loc='upper right', fontsize=15)
#     # plt.tight_layout()
#     # plt.savefig("results/dataset/pos_sample_count_variance.pdf", dpi=400)
# 
# 
#     # Doing this to check that the means increase with more positive samples
#     df = pd.read_csv(file_path, header=0, sep='\t', dtype=dtypes)
#     # Dropping bc7med because its a major outlier
#     # df.drop(df[df.dataset=='bc7med'].index, inplace=True)
#     mean_by_dataset = df.groupby(['dataset']).mean(numeric_only=True)
#     # THESE ARE ALPHABETIZED
#     means = mean_by_dataset['micro_f1_av'].tolist()
# 
#     popt, pcov = curve_fit(linear_func, np.log(sample_size_pos), means)
#     sample_residuals = means - linear_func(np.log(sample_size_pos), popt[0], popt[1])
#     sample_ss_res = np.sum(sample_residuals ** 2)
#     sample_ss_tot = np.sum((means - np.mean(means))**2)
# 
#     mean_r_squared = 1 - (sample_ss_res / sample_ss_tot)
# 
#     print("OPT params:", popt)
#     print("COVs:", pcov)
#     print("R^2:", mean_r_squared)
# 
#     # fig, ax = plt.subplots(figsize=[12,8])
# 
#     ax[0].set_xlabel("Log of number of positive samples in dataset", fontsize=15)
#     ax[0].set_ylabel("F1 mean", fontsize=15)
#     # ax[0].set_title("Mean micro f1 score vs positive sample size", fontsize=20)
#     ax[0].plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in np.log(sample_size_pos)], label=r"$r^{2}=$" + f"{mean_r_squared:.2f}", linewidth=5, color="navy", zorder=0)
#     ax[0].scatter(sample_size_pos, means, linewidth=5, color="deepskyblue", zorder=10)
#     handles, labels = ax[0].get_legend_handles_labels()
#     ax[0].legend(handles, labels, loc='upper left', fontsize=15)
#     ax[0].set_xscale('log')
#     fig.suptitle("Dataset scores versus number of positive samples", fontsize=20, y=0.92)
# 
#     plt.tight_layout(pad=3.0)
#     plt.savefig("results/dataset/pos_sample_plots.pdf", dpi=400)


def main_thesis():
    # THESE NEED TO BE ALPHABETIZED
    # ademiner, bc7dcpi, bc7med, cdr, cometa, i2c2, n2c2, ncbi, nlmchem
    # Includes all datasets
    dataset_word_size = [291345, 922104, 1481246, 271175, 463200, 364446, 904414, 146791, 729228]

    # Only looks at the number of positive samples 
    sample_size_pos = np.array([1300, 35020, 311, 11542, 19988, 22489, 7016, 3937, 16728])

    file_path = os.path.join("results", "experiment_1_results", "final_results_master.tsv")
    dtypes = {'dataset':str, 'lm_name':str, "micro_precision_av":float, "micro_precision_std":float, "micro_recall_av":float, "micro_recall_std":float, "micro_f1_av":float, "micro_f1_std":float, "macro_precision_av":float, "macro_precision_std":float, "macro_recall_av":float, "macro_recall_std":float, "macro_f1_av":float, "macro_f1_std":float}

    fig, ax = plt.subplots(figsize=[12,8])
    # Doing this to check that the means increase with more positive samples
    df = pd.read_csv(file_path, header=0, sep='\t', dtype=dtypes)
    mean_by_dataset = df.groupby(['dataset']).mean(numeric_only=True)
    # THESE ARE ALPHABETIZED
    means = mean_by_dataset['micro_f1_av'].tolist()
    print("Means", means)

    popt, pcov = curve_fit(linear_func, np.log(sample_size_pos), means)
    sample_residuals = means - linear_func(np.log(sample_size_pos), popt[0], popt[1])
    sample_ss_res = np.sum(sample_residuals ** 2)
    sample_ss_tot = np.sum((means - np.mean(means))**2)

    mean_r_squared = 1 - (sample_ss_res / sample_ss_tot)

    print("OPT params:", popt)
    print("COVs:", pcov)
    print("R^2:", mean_r_squared)

    # fig, ax = plt.subplots(figsize=[12,8])

    ax.set_xlabel("Log of number of positive samples in dataset", fontsize=15)
    ax.set_ylabel("F1 mean", fontsize=15)
    # ax[0].set_title("Mean micro f1 score vs positive sample size", fontsize=20)
    ax.plot(sample_size_pos, [linear_func(val, popt[0], popt[1]) for val in np.log(sample_size_pos)], label=r"$r^{2}=$" + f"{mean_r_squared:.2f}", linewidth=5, color="navy", zorder=0)
    ax.scatter(sample_size_pos, means, linewidth=5, color="deepskyblue", zorder=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=15)
    ax.set_xscale('log')
    fig.suptitle("Aggregated dataset scores versus number of positive samples", fontsize=20, y=0.92)

    plt.tight_layout(pad=3.0)
    plt.savefig("results/dataset/pos_sample_plots.pdf", dpi=400)

if __name__=="__main__":
    main_thesis()

