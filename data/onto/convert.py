import os
import re
import glob
import shutil
from pathlib import Path

CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
ONTO_CLASS_MAP = {NONE_CLASS:0, 'GPE': 1, 'ORDINAL': 2, 'DATE': 3, 'CARDINAL': 4, 'ORG': 5, 'PERCENT': 6, 'NORP': 7, 'MONEY': 8, 'PERSON': 9, 'LOC': 10, 'TIME': 11, 'WORK_OF_ART': 12, 'LAW': 13, 'QUANTITY': 14, 'EVENT': 15, 'PRODUCT': 16, 'FAC': 17, 'LANGUAGE': 18}

# This works, correct number of files found
def collect_all_ner_docs():
    new_directory = os.path.join("ner_only")
    ner_file_path_card = os.path.join("english", "annotations", "**", "*.name")

    num_files = 0
    Path(new_directory).mkdir(parents=True, exist_ok=True)
    for ner_file in glob.iglob(ner_file_path_card, recursive=True):
        num_files += 1

        shutil.copy2(ner_file, new_directory)
        new_ner_file = ner_file.split(os.sep)[-1]
        os.rename(os.path.join(new_directory, new_ner_file), os.path.join(new_directory, new_ner_file + f"{num_files}"))

    print(f"Moved {num_files} to {new_directory}")


def find_all_class_types():
    new_directory = os.path.join("ner_only")
    class_dict = {}
    len_tag = len('<ENAMEX TYPE="')
    for _, _, ner_files in os.walk(new_directory):
        for ner_file in ner_files:
            with open(os.path.join(new_directory, ner_file), "r+") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                matches = re.findall(r'<ENAMEX TYPE="[\w]+">', line)
                for match in matches:
                    entity_type = match[len_tag:-2]
                    class_dict[entity_type] = class_dict.get(entity_type, 0) + 1

    print("Class dictionary:")
    print(class_dict, "\n")
        

def convert_onto():
    new_directory = os.path.join("ner_only")
    len_tag = len('<ENAMEX TYPE="')
    end_tag = '</ENAMEX>'
    for _, _, ner_files in os.walk(new_directory):
        print("Num files to process:", len(ner_files))
        for ner_file in ner_files[:1]:
            with open(os.path.join(new_directory, ner_file), "r+") as f:
                lines = f.readlines()

            for line in lines[1:2]: 
                print(line)
                line = line.strip()
                if not line:
                    continue

                annotation = []
                matches = re.finditer(r'<ENAMEX TYPE="[\w]+">', line)
                num_removed = 0
                previous_index = 0
                for match in matches:
                    start, end = match.start() - num_removed, match.end() - num_removed

                    zeros_before = [0] * len(line[previous_index:start].split())
                    annotation.extend(zeros_before)

                    match_string = line[start: end]
                    entity_type = match_string[len_tag:-2]

                    # Need to add this to the annotations
                    ONTO_CLASS_MAP[entity_type]

                    line = line[:start] + line[end:]

                    end_tag_start = line.find(end_tag)
                    line = line[:end_tag_start] + line[end_tag_start + len(end_tag):]
                    
                    previous_index = end_tag_start + len(end_tag)
                    
                    num_removed += len(match_string) + len(end_tag)

                print(f"{line}\t{annotation}")


if __name__ == "__main__":
    collect_all_ner_docs()
    find_all_class_types()
    convert_onto()
