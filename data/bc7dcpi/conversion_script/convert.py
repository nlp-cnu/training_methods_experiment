import os
import spacy
import scispacy
import numpy as np
import re

CONVERTED_ALL_FILE = "converted_all.tsv"
CONVERTED_TRAIN_FILE = "converted_train.tsv"
CONVERTED_VAL_FILE = "converted_val.tsv"
CONVERTED_TEST_FILE = "converted_test.tsv"
NONE_CLASS = "none"
DCPI_CLASS_MAP = {NONE_CLASS:0, "CHEMICAL":1, "GENE-Y":2, "GENE-N":2, "GENE":2}

class Annotation:
    def __init__(self, text_id, entity_class, start, end, text):
        self.text_id = int(text_id)
        self.entity_class = entity_class
        self.start = int(start)
        self.end = int(end)
        self.text = text

class Document:

    def __init__(self, text_id, text):
        self.text_id = int(text_id)
        self.text = text
        self.annotations = []

def convert_DCPI():
    train_text_file = os.path.join("raw_data", "drugprot-gs-training-development/training/drugprot_training_abstracs.tsv")
    train_ann_file = os.path.join("raw_data", "drugprot-gs-training-development/training/drugprot_training_entities.tsv")
    dev_text_file = os.path.join("raw_data", "drugprot-gs-training-development/development/drugprot_development_abstracs.tsv")
    dev_ann_file = os.path.join("raw_data", "drugprot-gs-training-development/development/drugprot_development_entities.tsv")
    test_text_file = os.path.join("raw_data", "drugprot-gs-training-development/test-background/test_background_abstracts.tsv")
    test_ann_file = os.path.join("raw_data", "drugprot-gs-training-development/test-background/test_background_entities.tsv")

    # clear any existing files (since we append to them)
    if os.path.isfile(CONVERTED_ALL_FILE):
        os.remove(CONVERTED_ALL_FILE)
    if os.path.isfile(CONVERTED_TRAIN_FILE):
        os.remove(CONVERTED_TRAIN_FILE)
    if os.path.isfile(CONVERTED_VAL_FILE):
        os.remove(CONVERTED_VAL_FILE)
    if os.path.isfile(CONVERTED_TEST_FILE):
        os.remove(CONVERTED_TEST_FILE)

    process_set(train_text_file, train_ann_file, CONVERTED_ALL_FILE, CONVERTED_TRAIN_FILE)
    process_set(dev_text_file, dev_ann_file, CONVERTED_ALL_FILE, CONVERTED_VAL_FILE)
    process_set(test_text_file, test_ann_file, CONVERTED_ALL_FILE, CONVERTED_TEST_FILE)


def process_set(text_file, ann_file, all_output_file, individual_output_file):

    with open(text_file, "r+", encoding="utf-8") as f:
        titles_abstracts = f.readlines()
    with open(ann_file, "r+", encoding="utf-8") as f:
        annotations = f.readlines()

    text_ann_dict = {}
    for t in titles_abstracts:
        t = t.strip()
        text_id, title, abstract = t.split("\t")
        all_text = title + " " + abstract
        temp_doc = Document(text_id, all_text)
        text_ann_dict[text_id] = temp_doc

    for a in annotations:
        a = a.strip()
        text_id, entity_number, entity_class, char_start, char_end, text = a.split("\t")
        temp_ann = Annotation(text_id, entity_class, char_start, char_end, text)
        text_ann_dict[text_id].annotations.append(temp_ann)
            

    for text_id in text_ann_dict:
        doc = text_ann_dict[text_id]
        # Now process annotations with the text
        process_document(doc, DCPI_CLASS_MAP, all_output_file, individual_output_file)


def process_document(doc, class_map, all_output_file, individual_output_file):
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
        tokens = re.findall(r'\b\w+\b|[^\s\w]', line)
        #tokens = line.split()
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

        if not all_output_file is None:
            with open(all_output_file, 'a+', encoding='utf-8') as f: # Writing everything to file
                f.write(f"{line}\t{anns}\n")
        if not individual_output_file is None:
            with open(individual_output_file, 'a+', encoding='utf-8') as f: # Writing everything to file
                f.write(f"{line}\t{anns}\n")


if __name__ == "__main__":
    convert_DCPI()
