import os
import numpy as np
import json
import spacy
import scispacy
import re

CONVERTED_ALL_FILE = "converted_all.tsv"
CONVERTED_TRAIN_FILE = "converted_train.tsv"
CONVERTED_VAL_FILE = "converted_val.tsv"
CONVERTED_TEST_FILE = "converted_test.tsv"
NONE_CLASS = "none"
NLMCHEM_CLASS_MAP = {NONE_CLASS:0, "Chemical":1}

class Document:
    def __init__(self, text, text_id=None):
        self.text_id = text_id
        self.text = text
        self.annotations = []


class Annotation:
    def __init__(self, text_id, start, end, text, entity_class):
        self.text_id = text_id
        self.start = start
        self.end = end
        self.text = text
        self.entity_class = entity_class


def convert_NLM():
    train_file = os.path.join("raw_data", "BC7T2-NLMChem-corpus_v2.BioC.json", "BC7T2-NLMChem-corpus-train.BioC.json")
    dev_file = os.path.join("raw_data", "BC7T2-NLMChem-corpus_v2.BioC.json", "BC7T2-NLMChem-corpus-dev.BioC.json")
    test_file = os.path.join("raw_data", "BC7T2-NLMChem-corpus_v2.BioC.json", "BC7T2-NLMChem-corpus-test.BioC.json")

    # clear the output files (since we append to them)
    if os.path.isfile(CONVERTED_ALL_FILE):
        os.remove(CONVERTED_ALL_FILE)
    if os.path.isfile(CONVERTED_TRAIN_FILE):
        os.remove(CONVERTED_TRAIN_FILE)
    if os.path.isfile(CONVERTED_VAL_FILE):
        os.remove(CONVERTED_VAL_FILE)
    if os.path.isfile(CONVERTED_TEST_FILE):
        os.remove(CONVERTED_TEST_FILE)

    # read all the files
    with open(train_file, "r+", encoding='utf-8') as f:
        train = json.loads(f.read())
    with open(dev_file, "r+", encoding='utf-8') as f:
        dev = json.loads(f.read())
    with open(test_file, "r+", encoding='utf-8') as f:
        test = json.loads(f.read())

    # process the documents and group them
    train_docs = []
    val_docs = []
    test_docs = []
    for document in train["documents"]:
        for passage in document["passages"]:
            train_docs.append(convert_passage(passage))
    for document in dev["documents"]:
        for passage in document["passages"]:
            val_docs.append(convert_passage(passage))
    for document in test["documents"]:
        for passage in document["passages"]:
            test_docs.append(convert_passage(passage))

    class_map = NLMCHEM_CLASS_MAP
    for doc in train_docs:
        process_document(doc, CONVERTED_TRAIN_FILE, CONVERTED_ALL_FILE, class_map)
    for doc in val_docs:
        process_document(doc, CONVERTED_VAL_FILE, CONVERTED_ALL_FILE, class_map)
    for doc in test_docs:
        process_document(doc, CONVERTED_TEST_FILE, CONVERTED_ALL_FILE, class_map)


def convert_passage(passage):
    text = passage["text"].replace("\n", "")
    doc = Document(text)
    
    # Store passage information as Document structure
    root_offset = passage["offset"]

    for annotation in passage["annotations"]:
        entity_class = annotation["infons"]["type"]
        if entity_class == "Chemical":
            ann_id = annotation["id"]
            ann_text = annotation["text"]
            for loc in annotation["locations"]:
                o, l = int(loc["offset"]), int(loc["length"])
                start = o - root_offset
                end = start + l
                doc.annotations.append(Annotation(ann_id, start, end, ann_text, entity_class))

    return doc

def process_document(doc, individual_output_file, combined_output_file, class_map):
    nlp = spacy.load("en_core_sci_sm") # Using the small spacy model
    SENTENCE_INDEX = 0 # Sentences
    WS_INDEX = 1 # Sentences + whitespace
    sentences = [(s.text, s.text_with_ws) for s in nlp(doc.text).sents]
    annotated_lines = [[] for s in sentences]

    # There are sentences not seperated by spaces
    line_indexes = [0]
    for s, ws in sentences:
        sent_start = line_indexes[-1]
        sent_end = sent_start + len(s)
        amount_ws = len(ws) - len(s)
        next_sent_start = sent_end + amount_ws

        line_indexes.append(next_sent_start)

    for a in doc.annotations:
        index = 0
        while a.start >= line_indexes[index]:
            index += 1
        index = index - 1
        starting_char = a.start - line_indexes[index] 
        ending_char = a.end - line_indexes[index]
        annotated_lines[index].append((starting_char, ending_char, a.entity_class, sentences[index][SENTENCE_INDEX][starting_char:ending_char]))

    lines_and_annotations = list(zip([s for s, _ in sentences], annotated_lines)) # Now have all of the sentences and annotation character spans
    
    for line, annotations in lines_and_annotations: # Now to process the annotations for each token
        #tokens = line.split()
        tokens = re.findall(r'\b\w+\b|[^\s\w]', line)
        last_index = 0
        token_spans = []

        for t in tokens: # In case there are weird spaces between tokens
            t_start = line.find(t, last_index)
            t_end = t_start + len(t)
            last_index = t_end
            token_spans.append((t_start, t_end))

        anns = [class_map[NONE_CLASS] for _ in tokens] # Default is none/outside class

        for start, end, entity_type, text in annotations: # Mapping annotations to the correct tokens
            token_index = 0
            while start not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                token_index += 1
            start_index = token_index

            while token_index < len(token_spans) and end not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                token_index += 1

            end_index = token_index
            if token_index >= len(token_spans):
                # print("Annotation:", text)
                end_index -= 1


            for i in range(start_index, end_index + 1):
                anns[i] = class_map[entity_type]

        if anns != []:
            # write everything to file
            with open(individual_output_file, 'a+', encoding='utf-8') as of:
                of.write(f"{line}\t{anns}\n")
            with open(combined_output_file, 'a+', encoding='utf-8') as of:
                of.write(f"{line}\t{anns}\n")


if __name__ == "__main__":
    convert_NLM()


