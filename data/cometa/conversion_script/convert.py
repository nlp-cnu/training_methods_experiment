import os

import numpy as np
import pandas as pd
import re

COMETA_CLASS = "BiomedicalEntity"
CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
COMETA_CLASS_MAP = {NONE_CLASS:0, "BiomedicalEntity":1}

class Document:
    def __init__(self, text_id, text):
        self.text_id = int(text_id)
        self.text = text
        self.annotations = []


class Annotation:
    def __init__(self, text_id, text, start, end, entity_class):
        self.text_id = int(text_id)
        self.text = text
        self.start = int(start)
        self.end = int(end)
        self.entity_class = entity_class


def convert_cometa():
    train_file = os.path.join("raw_data", "cometa", "splits", "random", "train.csv")
    dev_file = os.path.join("raw_data", "cometa", "splits", "random", "dev.csv")
    test_file = os.path.join("raw_data", "cometa", "splits", "random", "test.csv")
    
    output_file = os.path.join(CONVERTED_DATASET_FILE)
    if os.path.isfile(output_file):
        os.remove(output_file)
    class_map = COMETA_CLASS_MAP

    train_samples = process_documents(train_file)
    dev_samples = process_documents(dev_file)
    test_samples = process_documents(test_file)

    documents = train_samples + dev_samples + test_samples
    process_all_samples(documents, output_file, class_map)
    
def process_documents(input_file):
    documents = []

    df = pd.read_csv(input_file, sep="\t", header=0)
    df = df[["ID", "Term", "Example"]]

    for row in df.iterrows():
        text_id, entity, text = row[1]["ID"], row[1]["Term"], row[1]["Example"]
        entity = entity.lower()

        if text is np.nan:
            continue
        temp_doc = Document(text_id, text)

        start = text.lower().find(entity)
        if start == -1:
            print(text_id)
            print(entity)
            continue
        end = start + len(entity)

        temp_doc.annotations.append(Annotation(text_id, entity, start, end, COMETA_CLASS))
        documents.append(temp_doc)

    return documents

def process_all_samples(documents, output_file, class_map):

    for doc in documents:
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

        # print(doc.text_id, doc.text, len(doc.text))

        for a in doc.annotations: # Mapping annotations to the correct tokens
            # print(a.text, a.start, a.end)

            start = a.start
            end = a.end

                    
            token_index = 0

            while start not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                token_index += 1
            start_index = token_index

            while token_index < len(token_spans) and end not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                token_index += 1
            end_index = token_index

            for i in range(start_index, end_index + 1):
                anns[i] = class_map[a.entity_class]

            with open(output_file, "a+", encoding="utf-8") as of:
                of.write(f"{doc.text}\t{anns}\n")




if __name__=="__main__":
    convert_cometa()
