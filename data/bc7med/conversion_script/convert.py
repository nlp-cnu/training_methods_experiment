import os
import numpy as np
import pandas as pd
import re

CONVERTED_DATASET_FILE = "converted.tsv"
NONE_CLASS = "none"
BC7MED_CLASS_MAP = {NONE_CLASS:0, "drug":1}

class Tweet:
    def __init__(self, text_id, text):
        self.tweet_id = text_id
        self.text = text
        self.annotations = []


class Annotation:
    def __init__(self, tweet_id, start, end, text):
        self.tweet_id = tweet_id
        self.start = int(start)
        self.end = int(end)
        self.text = text
        self.entity_type = "drug"


def convert_Med():
    train_file_a = os.path.join("raw_data", "BioCreative_TrainTask3.0.tsv")
    train_file_b = os.path.join("raw_data", "BioCreative_TrainTask3.1.tsv")
    dev_file = os.path.join("raw_data", "BioCreative_ValTask3.tsv")
    test_file = os.path.join("raw_data", "BioCreative_TEST_Task3_PARTICIPANTS.tsv")


    output_file = os.path.join(CONVERTED_DATASET_FILE)
    if os.path.isfile(output_file):
        os.remove(output_file)
    class_map = BC7MED_CLASS_MAP

    tweet_dict = {}

    with open(train_file_a, "r+", encoding="utf-8") as f:
        next(f)
        tweets = f.readlines()
    with open(train_file_b, "r+", encoding="utf-8") as f:
        next(f)
        tweets += f.readlines()
    with open(dev_file, "r+", encoding="utf-8") as f:
        next(f)
        tweets += f.readlines()
#    with open(test_file, "r+", encoding="utf-8") as f:
#        next(f)
#        tweets += f.readlines()
    

    for tweet in tweets:
        tweet = tweet.strip()
        tweet_id, user_id, created_at, text, start, end, span, drug = tweet.split("\t")
        
        entry = tweet_dict.get(tweet_id, Tweet(tweet_id, text))
        if drug != '-':
            entry.annotations.append(Annotation(tweet_id, start, end, span))
        tweet_dict[tweet_id] = entry
        
    for key in tweet_dict:
        tweet = tweet_dict[key]
        ann = make_annotations(tweet, class_map)
        with open(output_file, "a+", encoding="utf-8") as of:
            of.write(f"{tweet.text}\t{ann}\n")


def make_annotations(tweet, class_map): 
    #tokens = tweet.text.split()
    tokens = re.findall(r'\b\w+\b|[^\s\w]', tweet.text)
    last_index = 0
    token_spans = []

    for t in tokens: # In case there are weird spaces between tokens
        t_start = tweet.text.find(t, last_index)
        t_end = t_start + len(t)
        last_index = t_end
        token_spans.append((t_start, t_end))


    anns = [class_map[NONE_CLASS] for _ in tokens] # Default is none/outside class

    for a in tweet.annotations: # Mapping annotations to the correct tokens
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
            anns[i] = class_map[a.entity_type]

    return anns

if __name__ == "__main__":
    convert_Med()
