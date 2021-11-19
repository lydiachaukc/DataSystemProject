# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:58:56 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import random
import numpy as np
import time
import datetime

from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from classification_NN import classification_NN

from numbert2 import NumBert

from torch.utils.data import TensorDataset

def run_NumBert(trainset, validset, epochs, batch_size, lm, learning_rate):
    device = setup_cuda()
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    train_dataloader = prepare_data_loader(trainset, batch_size)
    valid_dataloader = prepare_data_loader(validset, batch_size)
    
    numbert_config = build_bert_config(
        trainset.combined_text_data.shape[1],
        trainset.combined_numeric_data.shape[1],
        lm)
    
    numBert_model = NumBert(numbert_config)

    optimizer = AdamW(numBert_model.parameters(),
      lr = learning_rate, 
      eps = 1e-8 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = len(train_dataloader) * epochs)
    '''
    Training NumBert
    '''
    numBert_model.train()
    
    total_t0 = time.time()

    total_train_loss = 0
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        epoch_t0 = time.time()
        
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = str(datetime.timedelta(seconds=int(round((time.time() - epoch_t0)))))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_numer_feats = batch[2].to(device)
            b_labels = batch[3].to(device)
    
            numBert_model.zero_grad()        
    
            result = numBert_model(
                numerical_features = b_numer_feats,
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels
                )
    
            loss = result['loss']
    
            total_train_loss += loss.item()
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(numBert_model.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
            
            print("training step:", step, " loss:", total_train_loss)
    
        avg_train_loss = total_train_loss / (len(train_dataloader) * batch_size)            
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        #print("  Training epcoh took: {:}".format(format_time(time.time() - epoch_t0)))
        
    #print("  Total training took: {:}".format(format_time(time.time() - total_t0)))
    
        '''
        Validating NumBert
        '''
        numBert_model.eval()
        
        total_valid_loss = 0
        
        for step, batch in enumerate(valid_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_numer_feats = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            with torch.no_grad():   
                result = numBert_model(
                    numerical_features = b_numer_feats,
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels
                    )
    
            total_valid_loss += result['loss'].item()
        
        avg_valid_loss = total_valid_loss / (len(valid_dataloader) * batch_size)        

        
def setup_cuda():
  if torch.cuda.is_available():    
      print('Running on GPU')
      return torch.device("cuda") 
  else:
      print('Running on CPU')
      return torch.device("cpu")

def prepare_data_loader(dataset,batch_size):
    tensor_dataset = TensorDataset(
            dataset.combined_text_data,
            dataset.text_attention_mask, 
            dataset.combined_numeric_data, 
            dataset.labels
            )
    
    return DataLoader(
        tensor_dataset,
        sampler = RandomSampler(tensor_dataset),
        batch_size = batch_size
    )
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def build_bert_config(text_input_dimension, num_input_dimension,lm):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
    )
    config.text_input_dimension = text_input_dimension
    config.num_input_dimension = num_input_dimension
    config.lm = lm
    return config
