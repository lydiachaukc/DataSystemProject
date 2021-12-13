# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:58:56 2021

@author: lydia
"""
import torch
import pandas as pd

from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertForSequenceClassification
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

from utils import setup_cuda, add_record


def train_valid_test_BertMatcher(trainset,
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
    output = pd.read_csv(output_directory + "/result.csv")
    today_date = str(pd.Timestamp.today().date())
    summary_writer = SummaryWriter(output_directory)
    summary_writer.add_text('Bert', 'Recording loss data for basic Bert Model', 0)
    
    device = setup_cuda()
    
    # Creating Dataloader
    train_dataloader = prepare_data_loader(trainset, batch_size)
    valid_dataloader = prepare_data_loader(validset, batch_size)
    test_dataloader = prepare_data_loader(testset, batch_size)
    
    # Creating BERT configuration
    bert_config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Get model and send it to CPU/GPU
    bert_model = BertForSequenceClassification.from_pretrained(lm, config = bert_config)
    bert_model.to(device)

    optimizer = AdamW(bert_model.parameters(),
      lr = learning_rate, 
      eps = 1e-8 
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = len(train_dataloader) * epochs)
    '''
    Training model
    '''
    bert_model.train()
        
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
                
        total_train_loss = 0
        #num_of_train_match = 0
        #num_of_valid_match = 0
        
        training_prediction = []
        training_labels = []
        validating_prediction = []
        validating_labels = []
        for step, batch in enumerate(train_dataloader):
  
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
            
            # recording result
            training_prediction += torch.max(result["logits"], dim = 1).indices.tolist()
            training_labels += b_labels.tolist()
            print("training step:",
                  step, "f1 score", 
                  f1_score(training_labels, training_prediction, zero_division=1, average="micro"))
            
            #loss_per_sample = loss.item()/ batch_size
            #accuracy = calculate_accuracy(b_labels, result['logits']).item()
            #num_of_train_match += accuracy * batch_size
            #print("training step:", step, " loss:", loss_per_sample, "accuracy:", accuracy)
        
        # recording result
        f1score = f1_score(training_labels, training_prediction, zero_division=1, average="micro")
        print("average training f1 score:", f1score)
        output = add_record(output, today_date, "numbert", 0, 0, f1score, "avg training f1", data_name)
        
        # number_of_sample = (len(train_dataloader) * batch_size)
        # avg_train_accuracy = num_of_train_match / number_of_sample
        # print("  Average training accuracy: {0:.5f}".format(avg_train_accuracy))
        # output = add_record(output, today_date, "numbert", 0, 0, avg_train_accuracy, "avg training accuracy", data_name)
        
        # avg_train_loss = total_train_loss / number_of_sample
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        #summary_writer.add_scalar("total training ", scalar_value = avg_train_loss , global_step = epoch)
        #output = add_record(output, today_date, "bert", 0, 0, avg_train_loss, "average training", data_name)
        
        '''
        Validating model
        '''
        bert_model.eval()
        
        total_valid_loss = 0
        
        for step, batch in enumerate(valid_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device) 
            b_labels = batch[2].to(device)
            b_input_segment = batch[3].to(device)
            
            with torch.no_grad():   
                result = bert_model(
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels,
                    token_type_ids  = b_input_segment
                    )
    
            total_valid_loss += result['loss'].item()
            
            # recording result
            validating_prediction += torch.max(result["logits"], dim = 1).indices.tolist()
            validating_labels += b_labels.tolist()
            #loss_per_sample = result['loss'].item() / batch_size
            # accuracy = calculate_accuracy(b_labels, result['logits']).item()
            # num_of_valid_match += accuracy * batch_size
            
            #print("validating step:", step, " loss:", loss_per_sample, "accuracy:", accuracy)
            
        # recording result
        f1score = f1_score(validating_labels, validating_prediction, zero_division=1, average="micro")
        print("average validating f1 score:", f1score)
        output = add_record(output, today_date, "numbert", 0, 0, f1score, "avg validating f1", data_name)
        
        # number_of_sample = (len(valid_dataloader) * batch_size)
        # avg_train_accuracy = num_of_valid_match / number_of_sample
        # print("  Average valid accuracy: {0:.5f}".format(avg_train_accuracy))
        # output = add_record(output, today_date, "bert", 0, 0, avg_train_accuracy, "avg training accuracy", data_name)

        # avg_valid_loss = total_valid_loss / number_of_sample
        # print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        # summary_writer.add_scalar("total_validating", scalar_value = avg_valid_loss , global_step = epoch)
        # output = add_record(output, today_date, "bert", 0, 0, avg_valid_loss, "average validating", data_name)
        
    '''
    Testing model
    '''
    bert_model.eval()
    
    total_test_loss = 0
    num_of_test_match = 0   
    testing_prediction = []
    testing_labels = [] 
    for step, batch in enumerate(test_dataloader):
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device) 
        b_labels = batch[2].to(device)
        b_input_segment = batch[3].to(device)
        
        with torch.no_grad():   
            result = bert_model(
                input_ids = b_input_ids,
                attention_mask = b_input_mask,
                labels = b_labels,
                token_type_ids  = b_input_segment
                )

        # total_test_loss += result['loss'].item()
        # accuracy = calculate_accuracy(b_labels, result['logits']).item()
        # num_of_test_match += accuracy * batch_size
        
        # recording loss result
        testing_prediction += torch.max(result["logits"], dim = 1).indices.tolist()
        testing_labels += b_labels.tolist()        
        #print("validation step:", step, " loss:", result['loss'].item() / batch_size, "accuracy:", accuracy)
        
    # recording loss result    
    f1score = f1_score(testing_labels, testing_prediction, zero_division=1, average="micro")
    print("average testing f1 score:", f1score)
    output = add_record(output, today_date, "numbert", 0, 0, f1score, "avg testing f1", data_name)
    # number_of_sample = (len(test_dataloader) * batch_size)
    # avg_test_accuracy = num_of_test_match / number_of_sample
    # print("  Average test accuracy: {0:.5f}".format(avg_test_accuracy))
    # output = add_record(output, today_date, "bert", 0, 0, avg_test_accuracy, "avg testing accuracy", data_name)

    # avg_test_loss = total_test_loss / number_of_sample
    # print("  Average test loss: {0:.2f}".format(avg_test_loss))
    # summary_writer.add_scalar("total_testing", scalar_value = avg_test_loss , global_step = epoch)
    # output = add_record(output, today_date, "bert", 0, 0, avg_test_loss, "average testing", data_name)
    
    output.to_csv(output_directory + "/result.csv", index=False)
    summary_writer.close()


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

def calculate_accuracy(actual_labels, logits):
    _, predicted_labels = torch.max(logits, dim = 1)    
    
    return (actual_labels == predicted_labels).sum().float() / len(actual_labels)