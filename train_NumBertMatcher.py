# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:58:56 2021

@author: lydia
"""
import pandas as pd
import torch
import random
import numpy as np
import time
import datetime as datetime
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tensorboardX import SummaryWriter

from numBertMatcher import NumBertMatcher

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

def train_and_valid_NumBertMatcher(trainset,
                                   validset,
                                   epochs,
                                   batch_size,
                                   lm,
                                   learning_rate,
                                   num_hidden_lyr,
                                   output_directory = "results",
                                   data_name = ""):
    
    output = pd.read_csv(output_directory + "/result.csv")
    device = setup_cuda()
    
    # setting seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    train_dataloader = prepare_data_loader(trainset, batch_size)
    valid_dataloader = prepare_data_loader(validset, batch_size)
    
    
    numbert_config = build_bert_config(
        trainset.combined_text_data.shape[1],
        trainset.numeric_dataA.shape[1],
        lm,
        num_hidden_lyr)
    
    
    numBertMatcher_model = NumBertMatcher.from_pretrained(lm_mp[lm], config = numbert_config)
    if torch.cuda.is_available(): 
        numBertMatcher_model.to(device)
        print("model to device")

    optimizer = AdamW(numBertMatcher_model.parameters(),
      lr = learning_rate, 
      eps = 1e-8 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = len(train_dataloader) * epochs)
    today_date = str(pd.Timestamp.today().date())
    summary_writer = SummaryWriter(output_directory + "/" + today_date)
    summary_writer.add_text('NumBerMatcher', 'Recording loss data for NumBerMatcher', 0)
    
    '''
    Training NumBert
    '''
    numBertMatcher_model.train()

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        epoch_t0 = time.time()
        
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = str(datetime.timedelta(seconds=int(round((time.time() - epoch_t0)))))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_numer_featsA = batch[2].to(device)
            b_numer_featsB = batch[3].to(device)
            b_labels = batch[4].to(device)
            b_input_segment = batch[5].to(device)
    
            numBertMatcher_model.zero_grad()        
    
            result = numBertMatcher_model(
                numerical_featuresA = b_numer_featsA,
                numerical_featuresB = b_numer_featsB,
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels,
                token_type_ids  = b_input_segment
                )
    
            loss = result['loss']
    
            total_train_loss += loss.item()
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(numBertMatcher_model.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
            
            # recording loss result
            loss_per_sample = loss.item()/ batch_size
            print("training step:", step, " loss:", loss_per_sample)
            summary_writer.add_scalar("training ", scalar_value = loss_per_sample , global_step = step)
            output = add_record(output, today_date, "numbert", epoch, step, loss_per_sample, "training", data_name)
            
        # recording loss result
        avg_train_loss = total_train_loss / (len(train_dataloader) * batch_size)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        summary_writer.add_scalar("total training ", scalar_value = avg_train_loss , global_step = epoch)
        output = add_record(output, today_date, "numbert", 0, 0, avg_train_loss, "average training", data_name)
    
        '''
        Validating NumBert
        '''
        numBertMatcher_model.eval()
        
        total_valid_loss = 0
        for step, batch in enumerate(valid_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_numer_featsA = batch[2].to(device)
            b_numer_featsB = batch[3].to(device)
            b_labels = batch[4].to(device)
            b_input_segment = batch[5].to(device)
            
            with torch.no_grad():   
                result = numBertMatcher_model(
                    numerical_featuresA = b_numer_featsA,
                    numerical_featuresB = b_numer_featsB,
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels,
                    token_type_ids  = b_input_segment
                    )
    
            total_valid_loss += result['loss'].item()
            
            loss_per_sample = result['loss'].item() / batch_size
            
            # recording loss result
            print("validation step:", step, " loss:", loss_per_sample)
            summary_writer.add_scalar("validating ", scalar_value = loss_per_sample , global_step = step)
            output = add_record(output, today_date, "numbert", epoch, step, loss_per_sample, "validation", data_name)
            
        # recording loss result
        avg_valid_loss = total_valid_loss / (len(valid_dataloader) * batch_size)
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        summary_writer.add_scalar("total validating ", scalar_value = avg_valid_loss , global_step = epoch)
        output = add_record(output, today_date, "numbert", 0, 0, avg_train_loss, "average validation", data_name)
        
    
    summary_writer.close()
    output.to_csv(output_directory + "/result.csv" , index=False)


def prepare_data_loader(dataset,batch_size):
    tensor_dataset = TensorDataset(
            dataset.combined_text_data,
            dataset.text_attention_mask, 
            dataset.numeric_dataA,
            dataset.numeric_dataB,
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


def build_bert_config(text_input_dimension, num_input_dimension, lm, num_hidden_lyr):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
    )
    config.text_input_dimension = config.hidden_size
    config.num_input_dimension = num_input_dimension
    config.num_hidden_lyr = num_hidden_lyr
    config.lm = lm
    return config
        
def setup_cuda():
  if torch.cuda.is_available():    
      print('Running on GPU')
      return torch.device("cuda") 
  else:
      print('Running on CPU')
      return torch.device("cpu")

def add_record(dataframe, time = "", model = "", epoch = "", batch = "", loss = "", purpose = "", data = ""):
    return dataframe.append(
        {"Time": time, "Model": model, "Epochs": epoch, "Batch": batch, "Loss": loss, "Purpose": purpose, "Data":data},
        ignore_index=True)