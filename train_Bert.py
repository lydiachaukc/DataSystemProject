# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:58:56 2021

@author: lydia
"""
import torch
import random
import numpy as np
import time
import datetime

from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}
def run_BertMatcher(trainset, validset, epochs, batch_size, lm, learning_rate, num_hidden_lyr):
    device = setup_cuda()
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    train_dataloader = prepare_data_loader(trainset, batch_size)
    valid_dataloader = prepare_data_loader(validset, batch_size)
    
    bert_config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    bert_model = BertForSequenceClassification.from_pretrained(lm_mp[lm], config = bert_config)
    if torch.cuda.is_available(): 
        bert_model.to(device)
        print("model to device")

    optimizer = AdamW(bert_model.parameters(),
      lr = learning_rate, 
      eps = 1e-8 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = len(train_dataloader) * epochs)
    '''
    Training NumBert
    '''
    bert_model.train()
    
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
            b_labels = batch[2].to(device)
            b_input_segment = batch[3].to(device)
    
            bert_model.zero_grad()        
    
            result = bert_model(
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels,
                token_type_ids  = b_input_segment
                )
    
            loss = result['loss']
    
            total_train_loss += loss.item()
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
            
            print("training step:", step, " loss:", loss.item())
    
        avg_train_loss = total_train_loss / (len(train_dataloader) * batch_size)            
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        #print("  Training epcoh took: {:}".format(format_time(time.time() - epoch_t0)))
        
    #print("  Total training took: {:}".format(format_time(time.time() - total_t0)))
    
        '''
        Validating NumBert
        '''
        bert_model.eval()
        
        total_valid_loss = 0
        
        for step, batch in enumerate(valid_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_labels = batch[2].to(device)
            
            with torch.no_grad():   
                result = bert_model(
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels
                    )
    
            total_valid_loss += result['loss'].item()
            print("validation step:", step, " loss:", result['loss'].item())
        avg_valid_loss = total_valid_loss / (len(valid_dataloader) * batch_size)        
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        
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
            dataset.labels,
            dataset.text_segment_ids
            )
    
    return DataLoader(
        tensor_dataset,
        sampler = RandomSampler(tensor_dataset),
        batch_size = batch_size,
        drop_last = True
    )
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
