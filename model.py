import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from utils import *


class FastDetectionDataset_Pred(Dataset):
    def __init__(self, config, questions, answers, max_length=512):
        self.config = config
        self.questions = questions
        self.answers = answers
        self.data = []
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def generate_instance(self, tokenizer, question_text, answer_text):
        q_tokens = roberta_tokenize(question_text, tokenizer, self.config.lowercase)

        sents = sentence_segment(answer_text)
        kept = []
        for s in sents:
            if not is_indicative(s):
                kept.append(s)

        answer_text = '. '.join(kept).strip()

        input_tokens = ['<s>'] + q_tokens + ['</s>', '</s>']
 
        input_tokens += roberta_tokenize(answer_text, tokenizer, self.config.lowercase)
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
            'attention_mask': attn_mask
        }

        return instance

    def numberize(self, tokenizer):
        data = []
        progress = tqdm.tqdm(total=len(self.answers), ncols=75,
                             desc='Numberizing')

        for question_text, answer_text in zip(self.questions, self.answers):
            progress.update(1)
            instance = self.generate_instance(tokenizer, question_text, answer_text)
            data.append(instance)

        progress.close()
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_attention_masks = []

        for inst in batch:
            batch_piece_idxs.append(inst['piece_idxs'])
            batch_attention_masks.append(inst['attention_mask'])

        batch_piece_idxs = torch.LongTensor(batch_piece_idxs).to(self.config.device)
        batch_attention_masks = torch.FloatTensor(batch_attention_masks).to(self.config.device)

        output_batch = []
        output_batch.append(batch_piece_idxs)
        output_batch.append(batch_attention_masks)
        return output_batch


class FastDetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocabs = vocabs

        self.bert = AutoModel.from_pretrained(config.bert_model_name,
                                                     cache_dir=config.bert_cache_dir,
                                                     output_hidden_states=True)
            
        self.bert_dim = self.bert.config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.detection_type_num = len(self.vocabs[config.dataset])

        self.detection_classifier_ffn = nn.Sequential(nn.Linear(self.bert_dim, 256), nn.ReLU(), nn.Linear(256, self.detection_type_num))
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def encode(self, piece_idxs, attention_mask):
        batch_size, _ = piece_idxs.size()
        bert_outputs = self.bert(piece_idxs, attention_mask=attention_mask)[0]
        cls_reprs = bert_outputs[:, 0, :]  
        cls_reprs = self.bert_dropout(cls_reprs)
        return cls_reprs

    def forward(self, batch):
        cls_reprs = self.encode(batch[0], batch[1])

        detection_scores = self.detection_classifier_ffn(cls_reprs) 
        loss = self.cross_entropy_loss(detection_scores.view(-1, self.detection_type_num), batch[2].view(-1))
        return loss, detection_scores

    def compute_detection_scores(self, questions, generated_answers, batch_size=32):
        assert len(questions) == len(generated_answers)

        eval_set = FastDetectionDataset_Pred(self.config, questions, generated_answers)
        eval_set.numberize(self.config.tokenizer)
        eval_batch_num = len(eval_set) // batch_size + \
                              (len(eval_set) % batch_size != 0)

        progress = tqdm.tqdm(total=eval_batch_num, ncols=75,
                             desc='Scoring')

        detection_scores = []
        for batch in DataLoader(eval_set, batch_size=batch_size,
                                shuffle=False, collate_fn=eval_set.collate_fn):
            progress.update(1)

            with torch.no_grad():
                cls_reprs = self.encode(batch[0], batch[1])

                scores = torch.softmax(self.detection_classifier_ffn(cls_reprs), dim=-1)[:, 1].data.cpu().numpy().tolist()  # [bs, ]

            detection_scores.extend(scores)

        assert len(detection_scores) == len(generated_answers)

        return detection_scores

