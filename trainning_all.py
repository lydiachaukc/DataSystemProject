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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="amazon_google")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--save_preprocessed_data", type=bool, default=True)
    parser.add_argument("--data_was_preprocessed", type=bool, default=False)
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--output_directory", type=str, default="results")
    parser.add_argument("--running_NumBertMatcher_crossencoder", type=bool, default=True)
    parser.add_argument("--running_NumBertMatcher_biencoder", type=bool, default=False)
    parser.add_argument("--running_BertMatcher", type=bool, default=False)
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
    
    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset_path = config['trainset']
    validset_path = config['validset']
    testset_path = config['testset']
    number_feature_columns = config['number_feature_columns']
    
    lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}
    lm = lm_mp[hp.lm]
    
    
    # preprocess all input data in datasetA and datasetB
    preprocessed_data = Load_and_preprocess(
        config,
        lm,
        number_feature_columns = number_feature_columns,
        data_was_preprocessed = hp.data_was_preprocessed,
        store_preprocessed_data = hp.save_preprocessed_data)
    
    
    '''
    Training and validating NumBertMatch crossencoder model
    '''
    if hp.running_NumBertMatcher_crossencoder:
        from train_NumBertMatcher_crossencoder import train_valid_test_NumBertMatcher_crossencoder
        
        # Build train/validate/test datasets for crossencoder
        trainset = build_tensor_dataset(preprocessed_data, trainset_path, is_cross_encoder=True)
        validset = build_tensor_dataset(preprocessed_data, validset_path, is_cross_encoder=True)
        testset = build_tensor_dataset(preprocessed_data, testset_path, is_cross_encoder=True)
        
        # Train, validate and test model
        train_valid_test_NumBertMatcher_crossencoder(
            trainset,
            validset,
            testset,
            epochs = hp.n_epochs,
            batch_size = hp.batch_size,
            lm = lm,
            learning_rate = hp.lr,
            num_hidden_lyr = 1)
    
    
    '''
    Training and validating NumBertMatch biencoder model
    '''
    if hp.running_NumBertMatcher_biencoder:
        from train_NumBertMatcher_bicoder import train_valid_test_NumBertMatcher_bicoder
        # Build train/validate/test datasets for biencoder
        trainset = build_tensor_dataset(preprocessed_data, trainset_path, is_cross_encoder=False)
        validset = build_tensor_dataset(preprocessed_data, validset_path, is_cross_encoder=False)
        testset = build_tensor_dataset(preprocessed_data, testset_path, is_cross_encoder=False)
        
        # Train, validate and test model
        train_valid_test_NumBertMatcher_bicoder(
            trainset,
            validset,
            testset,
            epochs = hp.n_epochs,
            batch_size = hp.batch_size,
            lm = lm,
            learning_rate = hp.lr,
            num_hidden_lyr = 2)
    
    
    '''
    Training and validating basic Bert model
    '''
    if hp.running_BertMatcher:
        from train_Bert import train_valid_test_BertMatcher
        # Preprocess all data again, so that numeric features are treated as text features
        preprocessed_data = Load_and_preprocess(
        config,
        lm,
        number_feature_columns = [],
        data_was_preprocessed = False,
        store_preprocessed_data = False)
    
        # Build train/validate/test datasets
        trainset = build_tensor_dataset(preprocessed_data, trainset_path)
        validset = build_tensor_dataset(preprocessed_data, validset_path)
        testset = build_tensor_dataset(preprocessed_data, testset_path)
        
        # Train, validate and test model
        train_valid_test_BertMatcher(
            trainset,
            validset,
            testset,
            epochs=hp.n_epochs,
            batch_size=hp.batch_size,
            lm = lm,
            learning_rate=hp.lr,
            num_hidden_lyr = 4,
            output_directory = hp.output_directory)