# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:58:56 2021

@author: lydia
"""
import pandas as pd
import torch
import time
import datetime as datetime
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tensorboardX import SummaryWriter

from utils import format_time, setup_cuda, add_record
from numBertMatcher import NumBertMatcher_crossencoder

def train_valid_test_NumBertMatcher_crossencoder(trainset,
                                   validset,
                                   testset,
                                   epochs,
                                   batch_size,
                                   lm,
                                   learning_rate,
                                   num_hidden_lyr,
                                   output_directory = "results",
                                   data_name = ""):
    # Set output format
    today_date = str(pd.Timestamp.today().date())
    summary_writer = SummaryWriter(output_directory + "/" + today_date)
    summary_writer.add_text('NumBerMatcher', 'Recording loss data for NumBerMatcher', 0)
    output = pd.read_csv(output_directory + "/result.csv")
    
    device = setup_cuda()
       
    # Creating Dataloader
    train_dataloader = prepare_data_loader(trainset, batch_size)
    valid_dataloader = prepare_data_loader(validset, batch_size)
    test_dataloader = prepare_data_loader(testset, batch_size)
    
    # Creating BERT configuration
    numbert_config = build_bert_config(
        trainset.numeric_dataA.shape[1],
        lm,
        num_hidden_lyr)
    
    # Get model and send it to CPU/GPU
    model = NumBertMatcher_crossencoder.from_pretrained(lm, config = numbert_config)
    model.to(device)

    optimizer = AdamW(model.parameters(),
      lr = learning_rate, 
      eps = 1e-8 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = len(train_dataloader) * epochs)
    
    '''
    Training model
    '''
    model.train()

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training crossencoder...')
        epoch_t0 = time.time()
        
        total_train_loss = 0
        num_of_train_match = 0
        num_of_valid_match = 0
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
    
            model.zero_grad()        
    
            result = model(
                numerical_featuresA = b_numer_featsA,
                numerical_featuresB = b_numer_featsB,
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels,
                token_type_ids  = b_input_segment
                )
    
            loss = result["loss"]
    
            total_train_loss += loss.item()
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
            
            # recording loss result
            loss_per_sample = loss.item() / batch_size
            num_of_train_match += result["accuracy"].item() *  batch_size
            print("training step:", step, " loss:", loss_per_sample, "accuracy: ", result["accuracy"].item())
            
            
        # recording loss result
        number_of_sample = (len(train_dataloader) * batch_size)
        avg_train_loss = total_train_loss / number_of_sample
        avg_train_accuracy = num_of_train_match / number_of_sample
        
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
        summary_writer.add_scalar("total training ", scalar_value = avg_train_loss , global_step = epoch)
        output = add_record(output, today_date, "numbert", 0, 0, avg_train_loss, "avg training loss", data_name)
        output = add_record(output, today_date, "numbert", 0, 0, avg_train_accuracy, "avg training accuracy", data_name)
    
    
        '''
        Validating model
        '''
        model.eval()
        
        total_valid_loss = 0
        for step, batch in enumerate(test_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_numer_featsA = batch[2].to(device)
            b_numer_featsB = batch[3].to(device)
            b_labels = batch[4].to(device)
            b_input_segment = batch[5].to(device)
            
            with torch.no_grad():   
                result = model(
                    numerical_featuresA = b_numer_featsA,
                    numerical_featuresB = b_numer_featsB,
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels,
                    token_type_ids  = b_input_segment
                    )
    
            total_valid_loss += result['loss'].item()
            
            # recording loss result
            loss_per_sample = result['loss'].item() / batch_size
            num_of_valid_match += result["accuracy"].item() *  batch_size
            print("validation step:", step, " loss:", loss_per_sample)
        
        
        # recording loss result
        number_of_sample = (len(valid_dataloader) * batch_size)
        avg_valid_loss = total_valid_loss / number_of_sample
        avg_valid_accuracy = num_of_valid_match / number_of_sample
        
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        print("  Average valid accuracy: {0:.2f}".format(avg_valid_accuracy))
        summary_writer.add_scalar("total training ", scalar_value = avg_valid_loss , global_step = epoch)
        output = add_record(output, today_date, "numbert", 0, 0, avg_valid_loss, "avg valid loss", data_name)
        output = add_record(output, today_date, "numbert", 0, 0, avg_valid_accuracy, "avg valid accuracy", data_name)
        
    
    '''
    Testing model
    '''
    model.eval()
    
    total_test_loss = 0
    num_of_test_match = 0
    for step, batch in enumerate(test_dataloader):
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device) 
        b_numer_featsA = batch[2].to(device)
        b_numer_featsB = batch[3].to(device)
        b_labels = batch[4].to(device)
        b_input_segment = batch[5].to(device)
        
        with torch.no_grad():   
            result = model(
                numerical_featuresA = b_numer_featsA,
                numerical_featuresB = b_numer_featsB,
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels,
                token_type_ids  = b_input_segment
                )

        total_test_loss += result['loss'].item()
        num_of_test_match += result["accuracy"].item() *  batch_size

    # recording loss result
    number_of_sample = (len(test_dataloader) * batch_size)
    avg_test_loss = total_test_loss / number_of_sample
    avg_test_accuracy = num_of_test_match / number_of_sample
    
    print("  Average test loss: {0:.2f}".format(avg_test_loss))
    print("  Average test accuracy: {0:.2f}".format(avg_test_accuracy))
    summary_writer.add_scalar("total training ", scalar_value = avg_valid_loss , global_step = epoch)
    output = add_record(output, today_date, "numbert", 0, 0, avg_test_loss, "avg test loss", data_name)
    output = add_record(output, today_date, "numbert", 0, 0, avg_test_accuracy, "avg test accuracy", data_name)

    summary_writer.close()
    output.to_csv(output_directory + "/result.csv" , index=False)


def prepare_data_loader(dataset, batch_size, random_sampler = True):
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

    
def build_bert_config(num_input_dimension, lm, num_hidden_lyr):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
    )
    config.text_input_dimension = config.hidden_size
    config.num_input_dimension = num_input_dimension
    config.num_hidden_lyr = num_hidden_lyr
    config.lm = lm
    return config