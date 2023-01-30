import os
import csv
import numpy as np

import spacy
import scispacy

CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
BC5CDR_CLASS_MAP = {NONE_CLASS:0, "Chemical":1, "Disease":2}


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


def convert_cdr():
    train_file = os.path.join("CDR_Data", "CDR.Corpus.v010516", "CDR_TrainingSet.PubTator.txt")
    dev_file = os.path.join("CDR_Data", "CDR.Corpus.v010516", "CDR_DevelopmentSet.PubTator.txt")
    test_file = os.path.join("CDR_Data", "CDR.Corpus.v010516", "CDR_TestSet.PubTator.txt")
    
    output_file = os.path.join(CONVERTED_DATASET_FILE)
    if os.path.isfile(output_file):
        os.remove(output_file)
    class_map = BC5CDR_CLASS_MAP

    train_samples = process_documents(train_file)
    dev_samples = process_documents(dev_file)
    test_samples = process_documents(test_file)

    documents = train_samples + dev_samples + test_samples
    process_all_samples(documents, output_file, class_map)


def process_documents(input_file):
    docs = []
    with open(input_file, "r+") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE) # title and abstract lines are pipe-delimited
        counter = 0 # Using this counter variable to find annotation lines
        title = None
        abstract = None
        temp_doc = None
        for row in reader:
            if counter == 0: # Dealing with the title
                sample_id, text_type, title = row
                counter += 1
                continue
            elif counter == 1: # Dealing with the abstract
                sample_id, text_type, abstract = row
                counter += 1
                temp_doc = Document(sample_id, title + " " + abstract)
                continue
            else:
                if row == []: # an empty line indicates that the annotations for the current article are done
                    counter = 0
                    title = None
                    abstract = None
                    docs.append(temp_doc)
                    temp_doc = None
                    continue

                if len(row[0].split('\t')) == 6:
                    sample_id, start, end, text, entity_class, mesh_id = row[0].split('\t') # The annotation lines are tab delimited
                    temp_doc.annotations.append((Annotation(sample_id, text, start, end, entity_class)))
                counter += 1
    return docs


def process_all_samples(documents, output_file, class_map):
    nlp = spacy.load('en_core_sci_sm') # Using this spacy model because it's quick to download and decent
    for doc in documents:
        sentences = [(s.text, s.text_with_ws) for s in nlp(doc.text).sents]
        annotated_lines = [[] for s in sentences]

        line_indexes = [0] # Adding spaces after each sentence except for the title
        for s, ws in sentences:
            sent_start = line_indexes[-1]
            sent_end = sent_start + len(s)
            amount_ws = len(ws) - len(s)
            next_sent_start = sent_end + amount_ws

            line_indexes.append(next_sent_start)
        

        for a in doc.annotations:
            index = 0
            while a.start >= line_indexes[index]: # Find line corresponding to this annotation
                index += 1

            index = index - 1

            starting_char = a.start - line_indexes[index]
            ending_char = a.end - line_indexes[index]
            annotated_lines[index].append((starting_char, ending_char, a.entity_class, sentences[index][0][starting_char:ending_char]))

        lines_and_annotations = list(zip(sentences, annotated_lines)) # Now have all of the sentences and annotation character spans

        
        for i, line_annotations in enumerate(lines_and_annotations): # Now to process the annotations for each token
            line, annotations = line_annotations
            line = line[0]
            tokens = line.split()
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

                while end not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                    token_index += 1
                end_index = token_index

                for i in range(start_index, end_index + 1):
                    anns[i] = class_map[entity_type]


            with open(output_file, 'a+') as of: # Writing everything to file
                of.write(f"{line}\t{anns}\n")


if __name__ == "__main__":
    convert_cdr()


