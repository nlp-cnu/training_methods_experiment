import os
import csv
import re

import numpy as np


CONVERTED_ALL_FILE = "converted_all.tsv"
CONVERTED_TRAIN_FILE = "converted_train.tsv"
CONVERTED_TEST_FILE = "converted_test.tsv"
NONE_CLASS = "none"
N2C2_CLASS_MAP = {NONE_CLASS:0, "Drug":1, "Strength":2, "Form":3, "Dosage":4, "Frequency":5, "Route":6, "Duration":7, "Reason":8, "ADE":9}
CHARACTER_LIM = 512


class Annotation:
    def __init__(self, text_id, entity_class, start, end, text):
        self.text_id = text_id
        self.entity_class = entity_class
        self.start = int(start)
        self.end = int(end)
        self.text = text


class Document:
    def __init__(self, text_id, text):
        self.text_id = text_id
        self.text = text
        self.annotations = []


def convert_n2c2():
    train_dir = os.path.join("raw_data", "training_20180910")
    test_dir = os.path.join("raw_data", "test")

    # clear the output files (since we append to them)
    if os.path.isfile(CONVERTED_ALL_FILE):
        os.remove(CONVERTED_ALL_FILE)
    if os.path.isfile(CONVERTED_TRAIN_FILE):
        os.remove(CONVERTED_TRAIN_FILE)
    if os.path.isfile(CONVERTED_TEST_FILE):
        os.remove(CONVERTED_TEST_FILE)

    # process and output the annotations
    process_set(train_dir, CONVERTED_TRAIN_FILE, CONVERTED_ALL_FILE)
    process_set(test_dir, CONVERTED_TEST_FILE, CONVERTED_ALL_FILE)

def process_set(text_dir, individual_output_file, combined_output_file):
    documents = []
    for _, _, files in os.walk(text_dir):
        for document in [f for f in files if ".ann" not in f]:
            with open(os.path.join(text_dir, document), "r+") as f: 
                text_id = document.split(".")[0]
                text = "".join([line for line in f.readlines()])
                temp_doc = Document(text_id, text)

            ann_file = text_id + ".ann"
            with open(os.path.join(text_dir, ann_file), "r+") as f:
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    if row[0][0] == "T":
                        ann_id, class_span, text = row[0], row[1], row[2]
                        class_span_items = class_span.split(" ")
                        entity_class, start, end = class_span_items[0], class_span_items[1], class_span_items[-1]
                        temp_doc.annotations.append(Annotation(ann_id, entity_class, start, end, text))
                        
            temp_doc.annotations.sort(key=lambda x: x.start)
            documents.append(temp_doc)

    for doc in documents:
        process_document(doc, N2C2_CLASS_MAP, individual_output_file, combined_output_file)


def process_document(doc, class_map, individual_output_file, combined_output_file):
    annotation_list = np.zeros(len(doc.text))
    spaces = [i for i, char in enumerate(doc.text) if char == " "]
    newlines = [i for i, char in enumerate(doc.text) if char == "\n"]
    annotation_list[spaces] = -1

    for ann in doc.annotations:
        annotation_list[ann.start: ann.end] = class_map[ann.entity_class]

    annotation_list = annotation_list.tolist()

    chunked_text = []
    chunked_annotations = []
    index = 0
    while index < len(doc.text) - CHARACTER_LIM:
        nearest_safe = index + CHARACTER_LIM
        while (annotation_list[nearest_safe] != -1):
            nearest_safe -= 1
        text = doc.text[index: nearest_safe]
        chunked_text.append(text)
        annotations = annotation_list[index: nearest_safe]
        chunked_annotations.append(annotations)
        index = nearest_safe + 1

    text = doc.text[index:]
    chunked_text.append(text)
    annotations = annotation_list[index:]
    chunked_annotations.append(annotations)

    recombined = " ".join(chunked_text)
    assert(recombined == doc.text)

    for line, annotation in zip(chunked_text, chunked_annotations):
        anns = []
        #words = line.split()
        words = re.findall(r'\b\w+\b|[^\s\w]', line)
        
        prev_index = 0
        for word in words:
            start = line.find(word, prev_index)
            end = start + len(word) - 1
            prev_index = end + 1
            anns.append(annotation[start])
        assert(len(words) == len(anns))

        line = line.replace("\n", " ")
        line = line.replace("\t", " ")
        line = re.sub(" +", " ", line)
        assert(len(words) == len(anns))

        with open(individual_output_file, 'a+', encoding='utf-8') as of: # Writing everything to file
            of.write(f"{line}\t{anns}\n")
        with open(combined_output_file, 'a+', encoding='utf-8') as of: # Writing everything to file
            of.write(f"{line}\t{anns}\n")


if __name__ == "__main__":
    convert_n2c2()
