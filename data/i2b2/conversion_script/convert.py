import os

import tempfile
import shutil

import pandas as pd
import numpy as np


CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
I2B2_CLASS_MAP = {NONE_CLASS:0, "problem":1, "treatment":2, "test":3}


class Document:
    def __init__(self, text, text_id=None):
        self.text_id = text_id
        self.text = text
        self.annotations = []


class Annotation:
    def __init__(self, line):
        # start and end are in the form of (line, word)
        self.text, self.start, self.end, self.entity_class = self.process_tag_line(line)

    def process_tag_line(self, line):
        concept_start_stop, tag = line.split("||")

        tag_start = tag.find('"') + 1
        tag_end = tag.rfind('"')
        tag = tag[tag_start: tag_end]

        concept_start = concept_start_stop.find('"') + 1
        concept_end = concept_start_stop.rfind('"')
        concept = concept_start_stop[concept_start: concept_end]

        start_start = concept_end + 2
        end_start = start_start + line[start_start:].index(' ')
        start = int(line[start_start:end_start].split(':')[0]), int(line[start_start:end_start].split(':')[1])

        start_stop = end_start + 1
        end_stop = start_stop + line[start_stop:].index('|')
        stop = int(line[start_stop:end_stop].split(':')[0]), int(line[start_stop:end_stop].split(':')[1])

        return concept, start, stop, tag


def convert_i2b2():
    train_data_dir_beth = os.path.join("raw_data", "concept_assertion_relation_training_data", "beth", "txt")
    train_ann_dir_beth = os.path.join("raw_data", "concept_assertion_relation_training_data", "beth", "concept")
    train_data_dir_partners = os.path.join("raw_data", "concept_assertion_relation_training_data", "partners", "txt")
    train_ann_dir_partners = os.path.join("raw_data", "concept_assertion_relation_training_data", "partners", "concept")
    test_data_dir = os.path.join("raw_data", "test_data", "txt")
    test_ann_dir = os.path.join("raw_data", "test_data", "concept")

    with tempfile.TemporaryDirectory() as tempdir:
        for src in [train_data_dir_beth, train_ann_dir_beth,
                train_data_dir_partners, train_ann_dir_partners,
                test_data_dir, test_ann_dir]:
            src_files = os.listdir(src)
            for file_name in src_files:
                full_file_name = os.path.join(src, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, tempdir)

        output_file = os.path.join(CONVERTED_DATASET_FILE)
        if os.path.isfile(output_file):
            os.remove(output_file)

        process_dataset(tempdir, output_file)


def process_dataset(text_dir, output_file):
    documents = []
    for _, _, files in os.walk(text_dir):
        for document in [f for f in files if ".con" not in f]:
            with open(os.path.join(text_dir, document), "r+") as f:
                text_id = document.split(".")[0]
                text = "".join([line for line in f.readlines()])
                temp_doc = Document(text, text_id)
            ann_file = text_id + ".con" 
            with open(os.path.join(text_dir, ann_file), "r+") as f:
                for line in [l.strip() for l in f.readlines()]:
                    temp_ann = Annotation(line)
                    temp_doc.annotations.append(temp_ann)
            documents.append(temp_doc)

    for doc in documents:
        process_document(doc, I2B2_CLASS_MAP, output_file)


def process_document(doc, class_map, output_file):
    text = doc.text
    annotations = doc.annotations
    annotations.sort(key=lambda x: (x.start[0], x.start[1]))
    ann_dict = {}
    for annotation in annotations:
        value = ann_dict.get(annotation.start[0], [])
        value.append(annotation)
        ann_dict[annotation.start[0]] = value

    for line_index, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        #words = line.split()
        words = re.findall(r'\b\w+\b|[^\s\w]', sample['text'])
        anns = np.zeros(len(words))
        cur_annotations = ann_dict.get(line_index, [])
        for ann in cur_annotations:
            class_ = class_map[ann.entity_class]
            start_index = ann.start[1]
            end_index = ann.end[1] + 1
            anns[start_index : end_index] = class_

        anns = anns.tolist()
        with open(output_file, 'a+') as of: # Writing everything to file
            of.write(f"{line}\t{anns}\n")


if __name__ == "__main__":
    convert_i2b2()

