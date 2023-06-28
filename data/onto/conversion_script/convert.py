import os
import re
import glob
import shutil
from pathlib import Path

CONVERTED_DATASET_FILE = "converted_all.tsv"
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
                matches = re.findall(r'<ENAMEX TYPE="[\w]+"(?: S_OFF="[\d]+")?(?: E_OFF="[\d]+")?>', line)
                for match in matches:
                    first_quote = match.find('"') + 1
                    second_quote = match.find('"', first_quote)
                    entity_type = match[first_quote: second_quote]
                    class_dict[entity_type] = class_dict.get(entity_type, 0) + 1

    print("Class dictionary:")
    print(class_dict, "\n")
    print("Total number of annotations:", sum([class_dict[key] for key in class_dict]))
        

def convert_onto():
    new_directory = os.path.join("ner_only")
    len_tag = len('<ENAMEX TYPE="')
    end_tag = '</ENAMEX>'
    for _, _, ner_files in os.walk(new_directory):
        print("Num files to process:", len(ner_files))
        for ner_file in ner_files:
            # print(ner_file)
            with open(os.path.join(new_directory, ner_file), "r+") as f:
                lines = f.readlines()

            for line in lines[1:-1]: 
                # print("Line:", line)
                line = line.strip()
                if not line:
                    continue

                try:
                    annotation = []
                    matches = re.finditer(r'<ENAMEX TYPE="[\w]+"(?: S_OFF="[\d]+")?(?: E_OFF="[\d]+")?>', line)
                    num_removed = 0
                    previous_index = 0
                    for match in matches:
                        start, end = match.start() - num_removed, match.end() - num_removed

                        zeros_before = [0] * (len(line[previous_index:start].strip().split()))
                        # print("Line before:", f"'{line[previous_index:start]}'")
                        # print("Num zeros before:", len(zeros_before))
                        annotation.extend(zeros_before)
                        # print("annotation after zeros:", annotation)

                        match_string = line[start: end]
                        first_quote = match_string.find('"') + 1
                        second_quote = match_string.find('"', first_quote)
                        entity_type = match_string[first_quote: second_quote]

                        # Need to add this to the annotations
                        class_type = ONTO_CLASS_MAP[entity_type]

                        line = line[:start] + line[end:]

                        end_tag_start = line.find(end_tag)
                        positive_labels = [class_type] * (len(line[start: end_tag_start].split()))
                        # print("start and end span:", start, end_tag_start)
                        # print("Between start and end span:", f"'{line[start: end_tag_start]}'")
                        # print("Num positives:", len(positive_labels))
                        annotation.extend(positive_labels)
                        # print("annotation after positive:", annotation)
                        line = line[:end_tag_start] + line[end_tag_start + len(end_tag):]
                        
                        previous_index = end_tag_start
                        # print("Previous index:", previous_index)
                        num_removed += len(match_string) + len(end_tag)
                        # print("Num removed:", num_removed)
                        # print()

                    zeros_after = [0] * (len(line[previous_index:].strip().split()))
                    annotation.extend(zeros_after)
                    # print(f"{line}\t{annotation}")
                    # print(f"{len(line.split())}\t{len(annotation)}")
                    if '">' in line or '<"' in line:
                        raise KeyError()
                    assert len(line.split()) == len(annotation), f"Error in {ner_file}, Line: {line}"
                    with open(CONVERTED_DATASET_FILE, 'a+') as of:
                        of.write(f"{line}\t{annotation}\n")
                except KeyError:
                    print(f"KeyError in {ner_file}, Line: {line}")


if __name__ == "__main__":
    # collect_all_ner_docs()
    find_all_class_types()
    convert_onto()
