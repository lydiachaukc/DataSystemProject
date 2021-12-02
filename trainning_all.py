# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:33:32 2021

@author: lydia
"""

import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from preprocess_data.load_and_preprocess import Load_and_preprocess
from build_dataset import build_tensor_dataset
from train_NumBertMatcher import train_and_valid_NumBertMatcher
from train_Bert import train_and_valid_BertMatcher

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="dirty_amazon_itunes")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--save_preprocessed_data", type=bool, default=True)
    parser.add_argument("--data_was_preprocessed", type=bool, default=True)
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--number_feature_columns", type=list, default=["Price"])
    parser.add_argument("--output_directory", type=str, default="results")

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset_path = config['trainset']
    validset_path = config['validset']
    testset_path = config['testset']
    number_feature_columns = config['number_feature_columns']
    
    # preprocess all input data in datasetA and datasetb
    preprocessed_data = Load_and_preprocess(
        config,
        hp.lm,
        number_feature_columns = number_feature_columns,
        data_was_preprocessed = hp.data_was_preprocessed,
        store_preprocessed_data = hp.save_preprocessed_data)
    
    # build train/dev/test datasets
    trainset = build_tensor_dataset(preprocessed_data, trainset_path)
    validset = build_tensor_dataset(preprocessed_data, validset_path)
    testset = build_tensor_dataset(preprocessed_data, testset_path)
    
    '''
    Training and validating NumBertMatch
    '''
    running_NumBertMatcher = False
    if running_NumBertMatcher:
        train_and_valid_NumBertMatcher(
            trainset,
            validset,
            epochs = hp.n_epochs,
            batch_size = hp.batch_size,
            lm = hp.lm,
            learning_rate = hp.lr,
            num_hidden_lyr = 2)
    
    '''
    Training and validating basic Bert model
    '''
    running_BertMatcher = True
    if running_BertMatcher:
        train_and_valid_BertMatcher(
            trainset,
            validset,
            epochs=hp.n_epochs,
            batch_size=hp.batch_size,
            lm=hp.lm,
            learning_rate=hp.lr,
            num_hidden_lyr = 4,
            output_directory = hp.output_directory)