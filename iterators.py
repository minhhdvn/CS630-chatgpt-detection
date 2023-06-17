import os
import copy
import itertools
import json, random
import numpy as np
import torch, tqdm
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from utils import *
from copy import deepcopy
import string


class FastDetectionDataset(Dataset):
    def __init__(self, config, path, max_length=256):
        self.config = config

        self.path = path
        self.raw = {}
        self.data = []
        self.max_length = max_length
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        with open(self.path) as f:
            self.raw = json.load(f)

        print('Loaded data for {} pairs from {}'.format(len(self.raw), self.path))

    def generate_instances(self, tokenizer, question_text, answer_text, label):
        instances = []
        sents = sentence_segment(answer_text)
        kept = []
        for s in sents:
            if not is_indicative(s):
                kept.append(s)

        answer_text = '. '.join(kept).strip()
        if len(answer_text.strip()) == 0:
            answer_text = 'ChatGPT response'

        q_tokens = roberta_tokenize(question_text, tokenizer)

        input_tokens = ['<s>'] + deepcopy(q_tokens) + ['</s>', '</s>']
        input_tokens += roberta_tokenize(answer_text, tokenizer)
        input_tokens = input_tokens[: self.max_length - 1]
        input_tokens += ['</s>']

        pieces = input_tokens

        piece_idxs = tokenizer.encode(pieces, add_special_tokens=False)

        piece_idxs[0] = tokenizer.cls_token_id

        piece_idxs_len = len(piece_idxs)
        piece_idxs_pad = self.max_length - piece_idxs_len

        piece_idxs = piece_idxs + [tokenizer.pad_token_id] * piece_idxs_pad
        attn_mask = [1] * piece_idxs_len + [0] * piece_idxs_pad

        instance = {
            'piece_idxs': piece_idxs,
            'attention_mask': attn_mask,
            'candidate_type_idxs': [label]
        }

        instances.append(instance)

        return instances

    def numberize(self, tokenizer):
        max_seq_length = 0
        data = []
        progress = tqdm.tqdm(total=len(self.raw), ncols=100,
                             desc='Numberizing {}'.format(self.path))

        for d in self.raw:
            progress.update(1)
            question_text = d['question']
            answer_text = d['answer']
            label = 1 if d['label'] == 'ChatGPT' else 0

            instances = self.generate_instances(tokenizer, question_text, answer_text, label)
            data += instances

        progress.close()
        self.data = data
        
    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_attention_masks = []
        batch_candidate_type_idxs = []

        for inst in batch:
            batch_piece_idxs.append(inst['piece_idxs'])
            batch_attention_masks.append(inst['attention_mask'])
            batch_candidate_type_idxs.append(inst['candidate_type_idxs'])


        batch_piece_idxs = torch.LongTensor(batch_piece_idxs).to(self.config.device)
        batch_attention_masks = torch.FloatTensor(batch_attention_masks).to(self.config.device)
        batch_candidate_type_idxs = torch.LongTensor(batch_candidate_type_idxs).to(self.config.device)

        output_batch = []
        output_batch.append(batch_piece_idxs)
        output_batch.append(batch_attention_masks)
        output_batch.append(batch_candidate_type_idxs)
        return output_batch
