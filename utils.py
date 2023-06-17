import os
import json, csv
import glob, re
from copy import deepcopy
import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

vocabs = {
    'hc3': {
        'Human': 0,
        'ChatGPT': 1
    },
}


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def compute_score(gold_file, all_predictions):
    gold_labels = []
    with open(gold_file) as f:
        data = json.load(f)

    gold_labels = [1 if d['label'] == 'ChatGPT' else 0 for d in data]

    predicted_labels = [1 if d['predicted_score'] > 0.5 else 0 for d in all_predictions]

    p = precision_score(gold_labels, predicted_labels)
    r = recall_score(gold_labels, predicted_labels)
    f1 = f1_score(gold_labels, predicted_labels)

    return {'p': p, 'r': r, 'f1': f1}

def sentence_segment(text):
    sents = re.split('[.?!]', text)
    return sents

def is_indicative(sentence):
    if 'AI assistant' in sentence or "I'm sorry to hear that" in sentence or "There're a few steps" in sentence or "Hmm" in sentence or "Nope" in sentence or "My view is" in sentence:
        return True
    return False

def roberta_tokenize(text, tokenizer, do_lower_case=1):
    if do_lower_case:
        text = text.lower()
    return tokenizer.tokenize(text)
