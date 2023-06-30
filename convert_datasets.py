import os

import pandas as pd

from data.ademiner.conversion_script.convert import convert_ade
from data.ncbi.conversion_script.convert import convert_NCBI
from data.n2c2.conversion_script.convert import convert_n2c2
from data.onto.conversion_script.convert import convert_onto
from data.i2b2.conversion_script.convert import convert_i2b2
from data.cdr.conversion_script.convert import convert_cdr
from data.bc7med.conversion_script.convert import convert_Med
from data.bc7dcpi.conversion_script.convert import convert_DCPI
from data.cometa.conversion_script.convert import convert_cometa
from data.nlmchem.conversion_script.convert import convert_NLM

from data.combined_datasets.n2c2_inter.collect_dataset import combine_for_n2c2
from data.combined_datasets.ncbi_inter.collect_dataset import combine_for_ncbi
from data.combined_datasets.bc7med_inter.collect_dataset import combine_for_bc7med
from data.combined_datasets.nlmchem_inter.collect_dataset import combine_for_nlmchem
from data.combined_datasets.ademiner_inter.collect_dataset import combine_for_ademiner

from utilities.constants import *

def convert_all():
    conversion_dict = {ONTO_DATA: convert_onto, 
            I2B2_DATA: convert_i2b2, 
            N2C2_DATA: convert_n2c2, 
            BC5CDR_DATA: convert_cdr, 
            DCPI_DATA: convert_DCPI, 
            NLMCHEM_DATA: convert_NLM, 
            NCBI_DATA: convert_NCBI, 
            BC7MED_DATA: convert_Med, 
            COMETA_DATA: convert_cometa, 
            ADEMINER_DATA: convert_ade}
    home_dir = os.getcwd()
    for directory in DOMAIN_SPECIFIC_DATASETS:
        print(f"Converting {directory.split(os.sep)[-1]}")
        os.chdir(os.path.join(home_dir, directory))
        conversion_dict[directory]()
        os.chdir(home_dir)


def create_minis():
    for dataset in DOMAIN_SPECIFIC_DATASETS:
        data_file = os.path.join(dataset, CONVERTED_DATASET_FILE)
        mini_data_file = os.path.join(dataset, CONVERTED_DATASET_FILE_MINI)
        df = pd.read_csv(data_file, sep='\t', header=None)
        mini_df = df.sample(20)
        mini_df.to_csv(mini_data_file, sep='\t', header=None, index=False)

def create_combined():

    conversion_dict = {ADEMINER_INTER_DATA: combine_for_ademiner,
                       BC7MED_INTER_DATA: combine_for_bc7med,
                       NLMCHEM_INTER_DATA: combine_for_nlmchem,
                       NCBI_INTER_DATA: combine_for_ncbi,
                       N2C2_INTER_DATA: combine_for_n2c2}
    home_dir = os.getcwd()

    for directory in EXP3B_INTER:
        print(f"Combining for {directory.split(os.sep)[-1]}")
        os.chdir(os.path.join(home_dir, directory))
        conversion_dict[directory]()
        os.chdir(home_dir)

if __name__ == "__main__":
    convert_all()
    #create_minis()
    create_combined()
