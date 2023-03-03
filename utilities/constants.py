import os

"""
This is a file that contains all of the constants used in my thesis experiments
"""
#########################################################################################
"""
These are all the dataset paths needed in these experiments
"""
ONTO_DATA = os.path.join("data", "onto")

I2B2_DATA = os.path.join("data", "i2b2")

N2C2_DATA = os.path.join("data", "n2c2")

BC5CDR_DATA = os.path.join("data", "cdr")

DCPI_DATA = os.path.join("data", "bc7dcpi")

NLMCHEM_DATA = os.path.join("data", "nlmchem")

NCBI_DATA = os.path.join("data", "ncbi")

BC7MED_DATA = os.path.join("data", "bc7med")

COMETA_DATA = os.path.join("data", "cometa")

ADEMINER_DATA = os.path.join("data", "ademiner")

ALL_DATASETS = [ONTO_DATA, I2B2_DATA, N2C2_DATA, BC5CDR_DATA, DCPI_DATA, NLMCHEM_DATA, NCBI_DATA, BC7MED_DATA, COMETA_DATA, ADEMINER_DATA]

DOMAIN_SPECIFIC_DATASETS = ALL_DATASETS[1:]

RESULTS_DIR_PATH = "results"

STATISTICS_DIR = os.path.join(RESULTS_DIR_PATH, "dataset")
STATISTICS_FILE = os.path.join(STATISTICS_DIR, "dataset_statistics.txt")
CONVERTED_DATASET_FILE = "converted.tsv"
CONVERTED_DATASET_FILE_MINI = "converted_mini.tsv"

EXPERIMENT_1_RESULTS = "experiment_1_results"
EXPERIMENT_2_RESULTS = "experiment_2_results"
EXPERIMENT_3_RESULTS = "experiment_3_results"

FINAL_RESULTS_FILE = "final_results.tsv"
FINAL_RESULTS_FILE_AMMIT = "final_results_ammit_exp1.tsv"

#########################################################################################
"""
These are all of the classes for each of the datasets.
"""
NONE_CLASS = "none"

ONTO_CLASS_MAP = {NONE_CLASS:0, 'GPE': 1, 'ORDINAL': 2, 'DATE': 3, 'CARDINAL': 4, 'ORG': 5, 'PERCENT': 6, 'NORP': 7, 'MONEY': 8, 'PERSON': 9, 'LOC': 10, 'TIME': 11, 'WORK_OF_ART': 12, 'LAW': 13, 'QUANTITY': 14, 'EVENT': 15, 'PRODUCT': 16, 'FAC': 17, 'LANGUAGE': 18}

I2B2_CLASS_MAP = {NONE_CLASS:0, "problem":1, "treatment":2, "test":3}

N2C2_CLASS_MAP = {NONE_CLASS:0, "Drug":1, "Strength":2, "Form":3, "Dosage":4, "Frequency":5, "Route":6, "Duration":7, "Reason":8, "ADE":9}

BC5CDR_CLASS_MAP = {NONE_CLASS:0, "Chemical":1, "Disease":2}

DCPI_CLASS_MAP = {NONE_CLASS:0, "CHEMICAL":1, "GENE-Y":2, "GENE-N":2, "GENE":2}

NLMCHEM_CLASS_MAP = {NONE_CLASS:0, "Chemical":1}

NCBI_CLASS_MAP = {NONE_CLASS:0, "Modifier":1, "SpecificDisease":2, "DiseaseClass":3, "CompositeMention":4}

BC7MED_CLASS_MAP = {NONE_CLASS:0, "drug":1}

COMETA_CLASS_MAP = {NONE_CLASS:0, "BiomedicalEntity":1}

ADEMINER_CLASS_MAP = {NONE_CLASS:0, "ADE":1}

DATASET_TO_CLASS_MAP = {"onto": ONTO_CLASS_MAP,
                        "i2b2": I2B2_CLASS_MAP,
                        "n2c2": N2C2_CLASS_MAP,
                        "cdr": BC5CDR_CLASS_MAP,
                        "bc7dcpi": DCPI_CLASS_MAP,
                        "nlmchem": NLMCHEM_CLASS_MAP,
                        "ncbi": NCBI_CLASS_MAP,
                        "bc7med": BC7MED_CLASS_MAP,
                        "cometa": COMETA_CLASS_MAP,
                        "ademiner": ADEMINER_CLASS_MAP}

#########################################################################################
"""
Here are all of the BERT models used in this experiment
"""

BASE_BERT = 'bert-base-uncased'

BIO_BERT = os.path.join('..', 'models', 'biobert_v1.1_pubmed')

PUBMED_BERT = os.path.join('..', 'models', 'BiomedNLP-PubMedBERT-base-uncased-abstract')

BIOCLINICAL_BERT = os.path.join('..', 'models', 'biobert_pretrain_output_all_notes_150000')

BIOCLINICAL_DISCHARGE_SUMMARY_BERT = os.path.join('..', 'models', 'biobert_pretrain_output_disch_100000')

BLUE_BERT_PUBMED = os.path.join('..', 'models', 'NCBI_BERT_pubmed_uncased_L-12_H-768_A-12')

BLUE_BERT_PUBMED_MIMIC = os.path.join('..', 'models', 'NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12')

BERTweet = os.path.join('..', 'models', 'bertweet-base')

BIO_REDDIT_BERT = os.path.join('..', 'models', 'BioRedditBERT-uncased')

ALL_MODELS = [BASE_BERT, BIO_BERT, PUBMED_BERT, BIOCLINICAL_BERT, BIOCLINICAL_DISCHARGE_SUMMARY_BERT, BLUE_BERT_PUBMED, BLUE_BERT_PUBMED_MIMIC, BERTweet, BIO_REDDIT_BERT]

DOMAIN_SPECIFIC_MODELS = ALL_MODELS[1:]

EXP1_WINNING_MODELS = [BLUE_BERT_PUBMED_MIMIC, BIOCLINICAL_DISCHARGE_SUMMARY_BERT, BLUE_BERT_PUBMED, BLUE_BERT_PUBMED, BIO_REDDIT_BERT, BLUE_BERT_PUBMED, BLUE_BERT_PUBMED, BIO_REDDIT_BERT, PUBMED_BERT]

########################################################################################
"""
Hyperparameters used by all models for all experiments. These should not change between experiments
"""

LEARNING_RATE = 1e-5
BATCH_SIZE = 16
MAX_EPOCHS = 20
MAX_NUM_TOKENS = 512
CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
SILENT = 0

SEED = 3
NUM_FOLDS = 5

# Want a 70:10:20 train-validation-test split
# CV splits all data into 80:20 train-test
# Taking 12.5% of the training data gets correct ratios 
VALIDATION_SIZE = 0.125
EARLY_STOPPING_PATIENCE = 5
########################################################################################
