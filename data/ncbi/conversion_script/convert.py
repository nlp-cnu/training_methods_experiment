import os
import csv
import spacy
import scispacy
import numpy as np
import re

CONVERTED_ALL_FILE = "converted_all.tsv"
CONVERTED_TRAIN_FILE = "converted_train.tsv"
CONVERTED_VAL_FILE = "converted_val.tsv"
CONVERTED_TEST_FILE = "converted_test.tsv"
NONE_CLASS = "none"
NCBI_CLASS_MAP = {NONE_CLASS:0, "Modifier":1, "SpecificDisease":2, "DiseaseClass":3, "CompositeMention":4}


# Final format is a csv file where we have line - annotation pairs
# Have to watch out for the fact that title, abstracts, and annotations are shoved into the files haphazardly

class Annotation:
    def __init__(self, start, end, text, entity_class):
        self.start = int(start)
        self.end = int(end)
        self.text = text
        self.entity_class = entity_class

    def __str__(self):
        return f"Text: {self.text}, Start: {self.start}, End: {self.end}"


def convert_NCBI():
    # ex_train = os.path.join("ex_train.txt")
    train_file = os.path.join("raw_data", "NCBItrainset_corpus.txt")
    dev_file = os.path.join("raw_data", "NCBIdevelopset_corpus.txt")
    test_file = os.path.join("raw_data", "NCBItestset_corpus.txt")

    # clear the output files (since we append to them)
    if os.path.isfile(CONVERTED_ALL_FILE):
        os.remove(CONVERTED_ALL_FILE)
    if os.path.isfile(CONVERTED_TRAIN_FILE):
        os.remove(CONVERTED_TRAIN_FILE)
    if os.path.isfile(CONVERTED_VAL_FILE):
        os.remove(CONVERTED_VAL_FILE)
    if os.path.isfile(CONVERTED_TEST_FILE):
        os.remove(CONVERTED_TEST_FILE)

    # process each file
    process_file(train_file, CONVERTED_TRAIN_FILE, CONVERTED_ALL_FILE)
    process_file(dev_file, CONVERTED_VAL_FILE, CONVERTED_ALL_FILE)
    process_file(test_file, CONVERTED_TEST_FILE, CONVERTED_ALL_FILE)

def process_file(input_file, individual_output_file, combined_output_file):
    class_map = NCBI_CLASS_MAP
    with open(input_file, "r+") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE) # title and abstract lines are pipe-delimited
        
        next(reader) # training file starts on a blank line
        
        counter = 0 # Using this counter variable to find annotation lines
        title = None
        abstract = None
        annotations = []
        for row in reader:
            if counter == 0 and len(row) == 0:
                # train and dev start with empty lines. Test does not
                # skip empty lines at the beginning of files 
                next
    
            if counter == 0: # Dealing with the title
                sample_id, text_type, title = row
                counter += 1
                continue
            elif counter == 1: # Dealing with the abstract
                sample_id, text_type, abstract = row
                counter += 1
                continue
            else:
                if row == []: # an empty line indicates that the annotations for the current article are done
                    process_title_abstract(title, abstract, annotations, individual_output_file, combined_output_file, class_map)

                    counter = 0
                    title = None
                    abstract = None
                    annotations = []
                    # break  # for development
                    continue

                sample_id, start, end, text, entity_class, mesh_id = row[0].split('\t') # The annotation lines are tab delimited
                annotations.append(Annotation(start, end, text, entity_class)) 
                counter += 1

        process_title_abstract(title, abstract, annotations, individual_output_file, combined_output_file, class_map) # The training file doesn't end with an empty line, so manually trigger this code

def process_title_abstract(title, abstract, annotations, individual_output_file, combined_output_file, class_map):
    all_text = title.strip() + " " + abstract.strip()

    # Using spacy to break into sentences
    nlp = spacy.load('en_core_sci_sm')
    sentences = [(s.text, s.text_with_ws) for s in nlp(all_text).sents]
    annotated_lines = [[] for s in sentences]
    
    line_indexes = [0] # Adding spaces after each sentence except for the title
    for s, ws in sentences:
        sent_start = line_indexes[-1]
        sent_end = sent_start + len(s)
        amount_ws = len(ws) - len(s)
        next_sent_start = sent_end + amount_ws
        line_indexes.append(next_sent_start)
    
    for a in annotations:
        index = 0
        while a.start >= line_indexes[index]: # Find line corresponding to this annotation
            index += 1

        index = index - 1

        starting_char = a.start - line_indexes[index]
        ending_char = a.end - line_indexes[index]
        annotated_lines[index].append((starting_char, ending_char, a.entity_class, sentences[index][0][starting_char:ending_char]))

    lines_and_annotations = list(zip(sentences, annotated_lines)) # Now have all of the sentences and annotation character spans

    
    for line, annotations in lines_and_annotations: # Now to process the annotations for each token
        line = line[0]
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
            while end not in range(token_spans[token_index][0], token_spans[token_index][1] + 1):
                # check if the span is longer than the sentence
                #  This happens if the sentence splitter splits a span (usually an error with splitting)
                #  truncate the span, and end it at the end of this sentence. This will essentially then,
                #  train on partial spans reducing final score
                if(token_index == len(token_spans)-1 and end >= token_spans[token_index][1]):
                    print (f"truncating span: {line}")
                    break
                token_index += 1
                    
            end_index = token_index

            for i in range(start_index, end_index + 1):
                anns[i] = class_map[entity_type]

        # write to the indivudal and combined files
        with open(individual_output_file, 'a+') as of:
            of.write(f"{line}\t{anns}\n")
        with open(combined_output_file, 'a+') as of:
            of.write(f"{line}\t{anns}\n")

if __name__ == "__main__":
    convert_NCBI()
