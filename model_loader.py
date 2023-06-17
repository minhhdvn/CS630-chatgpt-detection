import os
import json
import random
import time
from argparse import ArgumentParser
import logging
import sys
from datetime import datetime
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModel
from model import *
from iterators import *
from utils import *


# configuration
parser = ArgumentParser()
parser.add_argument('--bert_model_name', default='distilbert-base-uncased', type=str)
parser.add_argument('--bert_cache_dir', default='resource/bert', type=str)
parser.add_argument('--bert_dropout', default=0.15, type=float)
parser.add_argument('--dataset', default='hc3', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--eval_batch_size', default=64, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--max_subwords', default=256, type=int)
parser.add_argument('--lowercase', default=1, type=int)
parser.add_argument('--bert_learning_rate', default=1e-5, type=float)
parser.add_argument('--grad_clipping', default=5.0, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--output_dir', default='', type=str)
parser.add_argument('--train', default=0, type=int)
parser.add_argument('--device', default=0, type=int)

config = parser.parse_args()
    
os.environ['PYTHONHASHSEED'] = str(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if config.train:
    config.log_dir = 'logs'
    config.vocabs = vocabs

    if config.dataset == 'hc3':
        config.train_infile = 'datasets/hc3/train.json'
        config.dev_infile = 'datasets/hc3/dev.json'
        config.test_infile = 'datasets/hc3/test.json'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)

    # logging
    for name in logging.root.manager.loggerDict:
        if 'transformers' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=os.path.join(config.log_dir, '{}.training-log.txt'.format(config.dataset)),
                        filemode='w')
    global logger
    logger = logging.getLogger(__name__)

    # set GPU device
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_gpu = torch.cuda.device_count()

    # output
    best_model_fpath = os.path.join(config.log_dir, '{}.best-model.mdl'.format(config.dataset))
    # datasets
    model_name = config.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.bert_cache_dir)
    config.tokenizer = tokenizer

    # initialize the model
    model = FastDetectionModel(config)
    model.to(config.device)


    logger.info('================= Trainable params ({:.2f}M) ================='.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.))
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info('{}\t\t{}'.format(n, list(p.shape)))

    logger.info('==============================================================')

    # model state
    state = dict(model=model.state_dict())

    if config.n_gpu > 1 and not config.bert_model_name.startswith('preln'):
        model = torch.nn.DataParallel(model)

    train_set = FastDetectionDataset(config, config.train_infile, config.max_subwords)
    dev_set = FastDetectionDataset(config, config.dev_infile, config.max_subwords)
    test_set = FastDetectionDataset(config, config.test_infile, config.max_subwords)

    train_set.numberize(tokenizer)
    dev_set.numberize(tokenizer)
    test_set.numberize(tokenizer)

    batch_num = len(train_set) // config.batch_size
    dev_batch_num = len(dev_set) // config.eval_batch_size + \
                    (len(dev_set) % config.eval_batch_size != 0)
    test_batch_num = len(test_set) // config.eval_batch_size + \
                     (len(test_set) % config.eval_batch_size != 0)

    # optimizer
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters()],
            'lr': config.bert_learning_rate, 'weight_decay': 0
        },
    ]
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=batch_num * config.max_epoch)

    best_dev = {}
    best_test = {}
    for epoch in range(config.max_epoch):
        torch.cuda.empty_cache()
        print('Epoch: {}'.format(epoch))

        # training set
        progress = tqdm.tqdm(total=batch_num, ncols=75,
                             desc='Train {}'.format(epoch))
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(DataLoader(
                train_set, batch_size=config.batch_size,
                shuffle=True, collate_fn=train_set.collate_fn)):
            progress.update(1)
            loss, _ = model(batch)
            if config.n_gpu > 1:
                loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

        print('loss: {}'.format(loss.item()))
        progress.close()
        torch.cuda.empty_cache()
        model.eval()

        # dev set
        print('=' * 20)
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                             desc='Dev {}'.format(epoch))
        dev_preds = None
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            with torch.no_grad():
                _, logits = model(batch)

            if dev_preds is None:
                dev_preds = logits.detach().cpu().numpy()
            else:
                dev_preds = np.append(dev_preds, logits.detach().cpu().numpy(), axis=0)

        progress.close()

        preds_score = dev_preds.tolist()
        preds_score_softmax = []
        for item in preds_score:
            preds_score_softmax.append(np.exp(item[1]) / (np.exp(item[0]) + np.exp(item[1])))

        dev_preds = [{'predicted_score': s} for s in preds_score_softmax]

        dev_scores = compute_score(
            gold_file=config.dev_infile,
            all_predictions=dev_preds
        )
        print('Dev scores: {}'.format(dev_scores))
        # test set
        print('=' * 20)
        progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                             desc='Test {}'.format(epoch))
        test_preds = None
        for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            progress.update(1)
            with torch.no_grad():
                _, logits = model(batch)

            if test_preds is None:
                test_preds = logits.detach().cpu().numpy()
            else:
                test_preds = np.append(test_preds, logits.detach().cpu().numpy(), axis=0)

        progress.close()

        preds_score = test_preds.tolist()
        preds_score_softmax = []
        for item in preds_score:
            preds_score_softmax.append(np.exp(item[1]) / (np.exp(item[0]) + np.exp(item[1])))

        test_preds = [{'predicted_score': s} for s in preds_score_softmax]

        test_scores = compute_score(
            gold_file=config.test_infile,
            all_predictions=test_preds
        )
        print('Test scores: {}'.format(test_scores))

        if epoch == 0 or dev_scores['f1'] > best_dev['f1']:
            best_dev = dev_scores
            best_test = test_scores

            print('-' * 10)
            print('New best model: dev_scores = {}'.format(dev_scores))
            torch.save(state, best_model_fpath)

        print('=' * 20)
        print('Current best results:')
        print('Dev: Precision = {:.2f}, Recall = {:.2f}, F1 = {:.2f}'.format(best_dev['p'], best_dev['r'], best_dev['f1']))
        print('Test: Precision = {:.2f}, Recall = {:.2f}, F1 = {:.2f}'.format(best_test['p'], best_test['r'], best_test['f1']))
        print('-' * 10)


def load_detection_scorer(model_path, device=0):
    config.vocabs = vocabs

    # set GPU device
    config.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    config.n_gpu = 1

    # datasets
    model_name = config.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.bert_cache_dir)
    config.tokenizer = tokenizer

    # initialize the model
    detection_scorer = FastDetectionModel(config)
    detection_scorer.to(config.device)

    assert os.path.exists(model_path)
    model_weights = torch.load(model_path)['model']
    detection_scorer.load_state_dict(model_weights)
    detection_scorer.eval()
    return detection_scorer
