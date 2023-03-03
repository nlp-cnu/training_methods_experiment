from ast import literal_eval
import csv
import os
from pathlib import Path

from utilities.constants import *


def get_dataset_statistics():
    # Open all the datasets, count the number of words, 
    # count the number of words pertaining to a certain class
    # calculate the ratios,
    # write everything to a file

    try: # Fresh start every time
        os.remove(STATISTICS_FILE)
    except OSError:
        print()



    Path(STATISTICS_DIR).mkdir(parents=True, exist_ok=True)
    for dataset in ALL_DATASETS:
        try:
            num_pos_samples = 0
            dataset_path = os.path.join(dataset, CONVERTED_DATASET_FILE)
            dataset_name = dataset.split("/")[-1]

            class_map = DATASET_TO_CLASS_MAP[dataset_name] # annotation name -> index
            inverted_class_map = {} # index -> annotation name
            for key, value in class_map.items():
                inverted_class_map[value] = key
            
            sample_dict = {}
            count_dict = {}
            with open(dataset_path, "r+") as f:
                lines = f.readlines()
            
            for line in lines:
                text, annotation = line.split("\t")
                annotation = literal_eval(annotation) # converting from text to list
                for a in annotation:
                    if a > 0:
                        num_pos_samples += 1
                        break
                for key in class_map.values():
                    if key in annotation:
                        sample_dict[key] = sample_dict.get(key, 0) + 1
                reduced_annotation = []
                current_index = 0
                stopping_index = len(annotation)
                while current_index < stopping_index:
                    key = annotation[current_index]
                    if key == 0:
                        reduced_annotation.append(key)
                        current_index += 1
                    else:
                        reduced_annotation.append(key)
                        while current_index + 1 < stopping_index and key == annotation[current_index + 1]:
                            current_index += 1
                        current_index += 1
                for key in reduced_annotation:
                # for key in annotation:
                    count_dict[key] = count_dict.get(key, 0) + 1

            total_words = sum(count_dict.values())
            num_samples = len(lines)
            # Class ratio: # annotations / # total words
            ratio_count_dict = {key:value/total_words for key, value in count_dict.items()}        
            ratio_sample_dict = {key:value/num_samples for key, value in sample_dict.items()}        

            with open(STATISTICS_FILE, "a+") as f:
                f.write(f"{dataset_name} statistics:\n")
                f.write(f"annotation type\t# annotations\tratio\n")
                f.write(f"Total number of words: {total_words}\n")
                for key in count_dict:
                    annotation_type = inverted_class_map[key]
                    num_anns = count_dict[key]
                    ratio = ratio_count_dict[key]
                    f.write(f"{annotation_type}\t{num_anns}\t{ratio}\n")

                f.write("/\\"*25 + "\n")
                f.write(f"Total number of samples: {num_samples}\n")
                for key in sample_dict:
                    annotation_type = inverted_class_map[key]
                    num_samples = sample_dict[key]
                    ratio = ratio_sample_dict[key]
                    f.write(f"{annotation_type}\t{num_samples}\t{ratio}\n")
                f.write("-"*50 + "\n")

            print(f"Dataset={dataset}, num_pos_samples={num_pos_samples}")

        except FileNotFoundError:
            print(dataset + " not found")

if __name__ == "__main__":
    get_dataset_statistics()

