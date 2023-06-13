#!/usr/bin/env python
import logging as log
import sys
import os
from os import path
import pandas as pd
from pandas.core import series

class Tweet(object):
    '''Class for storing tweet'''
    def __init__(self, twid, text=""):
        self.twid = twid
        self.text = text
        self.has_drug = False
        self.anns = []


class Ann(object):
    '''Class for storing annotation spans'''
    def __init__(self, drg, start, end):
        self.drg = drg
        self.start = int(start)
        self.end = int(end)
        assert self.start<=self.end, "I found a tweet with an annotation where the starting position {} is higher than the ending position {}, check the data.".format(self.start, self.end)
    def __str__(self):
        return "{} => {}, {}".format(self.drg, self.start, self.end)


def load_dataset(gfile):
    """
    :param gfile: the path to the annotation file, a tsv containing tweet_id, user_id, created_at, text, start, end, span, drug
    :return: a dictionary tweet_id => tweet object (with their list of expected annotations)
    """
    tw_int_map = {}
    df = pd.read_csv(gfile, sep='\t')
    #def createTweets(tw:series, tw_int_map:dict):
    def createTweets(tw, tw_int_map):
        #create the tweet or retrieve the tweet if the tweet contains multiple drugs
        if tw['tweet_id'] in tw_int_map:
            tweet = tw_int_map[tw['tweet_id']]
        else:
            tweet = Tweet(tw['tweet_id'], tw['text'])
            tw_int_map[tw['tweet_id']] = tweet
        #add the annotations if there are
        if tw['span']!='-':
            ann = Ann(tw['span'].strip(), tw['start'], tw['end'])
            tweet.anns.append(ann)
    df.apply(lambda tw: createTweets(tw, tw_int_map), axis=1)
    num_anns = sum([len(x.anns) for _, x in tw_int_map.items()])
    #log.info("Loaded dataset %s tweets. %s annotations.", len(tw_int_map), num_anns)
    return tw_int_map        


def is_overlap_match(a, b):
    return b.start <= a.start <= b.end or a.start <= b.start <= a.end


def is_strict_match(a, b):
    return a.start == b.start and a.end == b.end


def is_match(a, b, strict):
    return is_strict_match(a, b) if strict else is_overlap_match(a, b)


def perf(gold_ds, pred_ds, strict=True):
    """Calculates performance and returns P, R, F1
    Arguments:
        gold_ds {dict} -- dict contaning gold dataset
        pred_ds {dict} -- dict containing prediction dataset
        strict {boolean} -- boolean indication if strict evaluation is to be used
    """
    g_tp, g_fn = [], []
    # find true positives and false negatives
    for gold_id, gold_tw in gold_ds.items():
        gold_anns = gold_tw.anns
        pred_anns = pred_ds[gold_id].anns
        for g in gold_anns:
            g_found = False
            for p in pred_anns:
                if is_match(p, g, strict):
                    g_tp.append(p)
                    g_found = True
            if not g_found:
                g_fn.append(g)
    p_tp, p_fp = [], []
    # find true positives and false positives
    for pred_id, pred_tw in pred_ds.items():
        pred_anns = pred_tw.anns
        gold_anns = gold_ds[pred_id].anns
        for p in pred_anns:
            p_found = False
            for g in gold_anns:
                if is_match(p, g, strict):
                    p_tp.append(p)
                    p_found = True
            if not p_found:
                p_fp.append(p)
    # both true positive lists should be same
    if len(g_tp) != len(p_tp):
        log.warn("Error: True Positives don't match. %s != %s", g_tp, p_tp)
    #log.info("AVIRER! TP:%s FP:%s FN:%s", len(g_tp), len(p_fp), len(g_fn))
    # now calculate p, r, f1
    precision = 1.0 * len(g_tp)/(len(g_tp) + len(p_fp) + 0.000001)
    recall = 1.0 * len(g_tp)/(len(g_tp) + len(g_fn) + 0.000001)
    f1sc = 2.0 * precision * recall / (precision + recall + 0.000001)
    if strict:
        log.info("Strict: Precision:%.3f Recall:%.3f F1:%.3f", precision, recall, f1sc)
    else:
        log.info("Overlapping: Precision:%.3f Recall:%.3f F1:%.3f", precision, recall, f1sc)
    return precision, recall, f1sc


def score_task(pred_file, gold_file, out_file):
    """Score the predictions and print scores to files
    Arguments:
        pred_file {string} -- path to the predictions file
        gold_file {string} -- path to the gold annotation file
        out_file {string} -- path to the file to write results to
    """
    # load gold dataset
    gold_ds = load_dataset(gold_file)
    # load prediction dataset
    pred_ds = load_dataset(pred_file)
        
    #Sanity check that the tweets are the same and the the texts of the tweets are also the same
    assert len(gold_ds)==len(pred_ds), "The number of tweets loaded in the gold standard {} is not the same than the number of tweets loaded in the predictions {}".format(len(gold_ds), len(pred_ds))
    for gtwID, gtw in gold_ds.items():
        assert gtwID in pred_ds.keys(), "The tweet {} of the gold standard was not found in the tweets of the predictions...".format(gtwID)
        ptw = pred_ds[gtwID]
        #for some reasons we may have a weird character at the end of the tweets, just remove them case by case... 
        if ord(gtw.text[-1])==65532:
            gtw.text=gtw.text[:-1]
        if ord(ptw.text[-1])==65532:
            ptw.text=ptw.text[:-1]
        assert gtw.text==ptw.text, "The text of the tweet {}:[{}] in the gold standard is different from the text of the same tweet {}:[{}] in the predictions...".format(gtwID, gtw.text, ptw.twid, ptw.text)
    
    out = open(out_file, 'w')
    o_prec, o_rec, o_f1 = perf(gold_ds, pred_ds, strict=False)
    out.write("Task3overlapF:%.3f\n" % o_f1)
    out.write("Task3overlapP:%.3f\n" % o_prec)
    out.write("Task3overlapR:%.3f\n" % o_rec)
    s_prec, s_rec, s_f1 = perf(gold_ds, pred_ds, strict=True)
    out.write("Task3strictF:%.3f\n" % s_f1)
    out.write("Task3strictP:%.3f\n" % s_prec)
    out.write("Task3strictR:%.3f\n" % s_rec)
    out.flush()


def evaluate():
    """
        Runs the evaluation function
        Expects the file ref/BioCreative_GoldStandardTask3.tsv as gold standard
        Expect one file in res/, does not check the name of the file but it should have the format expected.
        Write logs in BioCreative_Eval.log
    """
    # load logger
    #LOG_FILE = '/Users/dweissen/tmp/BioCreative_Eval.log'
    LOG_FILE = '/tmp/BioCreative20Task3_Eval.log'
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])
    # as per the metadata file, input and output directories are the arguments
    if len(sys.argv) != 3:
        log.error("Invalid input parameters. Format:\
                  \n python evaluation.py [input_dir] [output_dir]")
        sys.exit(0)
    [_, input_dir, output_dir] = sys.argv

    # get files in prediction zip file
    pred_dir = os.path.join(input_dir, 'res')
    pred_files = [x for x in os.listdir(pred_dir) if not os.path.isdir(os.path.join(pred_dir, x))]
    pred_files = [x for x in pred_files if x[0] not in ["_", "."]]
    if not pred_files:
        log.error("No valid files found in archive. \
                  \nMake sure file names do not start with . or _ characters")
        sys.exit(0)
    if len(pred_files) > 1:
        log.error("More than one valid files found in archive. \
                  \nMake sure only one valid file is available.")
        sys.exit(0)
    # Get path to the prediction file
    pred_file = os.path.join(pred_dir, pred_files[0])

    # Get path to the gold standard annotation file and score file
    if path.exists(os.path.join(input_dir, 'ref/BioCreative_ValTask3.tsv')):
        gold_file = os.path.join(input_dir, 'ref/BioCreative_ValTask3.tsv')
    else:
        if path.exists(os.path.join(input_dir, 'ref/BioCreative_TestTask3.tsv')):
            gold_file = os.path.join(input_dir, 'ref/BioCreative_TestTask3.tsv')
        else:
            log.error("Could not find the goldstandard file in the ref directory.")
            sys.exit(0)
    log.info("Pred file:%s, Gold file:%s", pred_file, gold_file)
    out_file = os.path.join(output_dir, 'scores.txt')
    log.info("Output file:%s", out_file)

    log.info("Start scoring")
    score_task(pred_file, gold_file, out_file)


    log.info("Finished scoring")

if __name__ == '__main__':
    evaluate()

