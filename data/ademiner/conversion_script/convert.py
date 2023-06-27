import os
import sys
import numpy as np
import re

CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
ADEMINER_CLASS_MAP = {NONE_CLASS:0, "ADE":1}

class Document:
    def __init__(self, text, text_id=None):
        self.text_id = text_id
        self.text = text
        self.annotations = []


class Annotation:
    def __init__(self, text_id, start, end, text, entity_class):
        self.text_id = text_id
        self.start = int(start)
        self.end = int(end)
        self.text = text
        self.entity_class = entity_class


def convert_ade():
    train_file = os.path.join("raw_data", "HLP-ADE-v1", "train_tweets.tsv")
    train_anns = os.path.join("raw_data", "HLP-ADE-v1", "train_annotations.tsv")

    output_file = os.path.join(CONVERTED_DATASET_FILE)
    if os.path.isfile(output_file):
        os.remove(output_file)
    class_map = ADEMINER_CLASS_MAP

    with open(train_file, "r+", encoding="utf-8") as f:
        tweets = [l.strip() for l in f.readlines()]
    with open(train_anns, "r+", encoding="utf-8") as f:
        anns = [l.strip() for l in f.readlines()]
    
    documents = {}

    for t in tweets:
        text_id, text = t.split("\t")
        documents[text_id] = Document(text, text_id=text_id)

    for a in anns:
        text_id, entity_class, start, end, text, _, _ = a.split("\t")
        documents[text_id].annotations.append(Annotation(text_id, start, end, text, entity_class))

    for doc_id in documents:
        doc = documents[doc_id]
        annotations = process_annotations(doc, class_map)
        with open(output_file, "a+", encoding="utf-8") as of:
            of.write(f"{doc.text}\t{annotations}\n")

def process_annotations(doc, class_map):
    #tokens = doc.text.split()
    tokens = re.findall(r'\b\w+\b|[^\s\w]', doc.text)
    last_index = 0
    token_spans = []

    for t in tokens: # In case there are weird spaces between tokens
        t_start = doc.text.find(t, last_index)
        t_end = t_start + len(t)
        last_index = t_end
        token_spans.append((t_start, t_end))

    anns = [class_map[NONE_CLASS] for _ in tokens] # Default is none/outside class
    
    for a in doc.annotations: # Mapping annotations to the correct tokens
        start = a.start
        end = a.end
                
        token_index = 0

        while start not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
            token_index += 1
        start_index = token_index

        while end not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
            token_index += 1
        end_index = token_index

        for i in range(start_index, end_index + 1):
            anns[i] = class_map[a.entity_class]

    return anns


if __name__ == "__main__":
    convert_ade()
