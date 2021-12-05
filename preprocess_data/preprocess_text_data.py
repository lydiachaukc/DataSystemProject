# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:29:32 2021

@author: lydia
"""
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch


stopwords = set(stopwords.words('english'))
# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

class Preprocess_text_data:
    """
    Convert text data to tokens that can be used in the entity matching models
    """
    def __init__(self,task_config, lm):
        self.config = task_config
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])

    def tokenize_dataset(self, dataset, insert_column_name = True,max_len=256):
        '''
        Convert a dataset of text data to a dataset of token ids

        '''
        tokenized_dataset =  pd.DataFrame(np.zeros((len(dataset),1)), columns=["len"])
        tokenized_dataset["all_text_data"] = np.empty((len(dataset), 0)).tolist()
        
        for col in dataset:
            tokenized_dataset[col] = None
            print("col:", col)
            tokenized_dataset["all_text_data"] += dataset[col].apply(
                lambda elem: self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(self.tokenize_field(elem)), col))
        tokenized_dataset["len"] = tokenized_dataset["all_text_data"].apply(lambda elem: len(elem))
        
        return tokenized_dataset[["all_text_data","len"]]
        
    
    def tokenize_field(self, sentence, header = None):
        '''
        Convert a sentence to list of tokens (string) and remove stopwords
        '''
        tokens = ""
        if header is not None:
            tokens = header + " [SEP] "
        
        words = sentence.split(' ')
        for word in words:
            if word not in stopwords:
                tokens += word + " [SEP] "
        return tokens
    
    
    def build_datset_for_bi_encoder(self, textdata, segment_id = 0):
        tokens, segment_ids = self.create_tensor_for_single_textdata(textdata, segment_id)
        attention_mask = self.build_attention_mask(tokens)
        return tokens, attention_mask, segment_ids
    
    
    def build_datset_for_cross_encoder(self, textdataA, textdataB):
        tokens = self.create_tensor_for_paired_textdata(textdataA, textdataB)
        attention_mask = self.build_attention_mask(tokens)
        return tokens, attention_mask
    
    
    def create_tensor_for_single_textdata(self, textdata, segment_id, max_len=512, label_token_len=3):
        '''
        Convert the dataset of token ids to an equal-width(512 tokens) tensor with zero padding on the right-hand-side
        Also create the segment id tensor

        '''
        max_text_len =  max_len - label_token_len
        tokens = textdata["all_text_data"]
        tokens.apply(lambda text: text[:max_text_len]if len(text) > max_text_len else text)
         
        if (len(tokens.iloc[0])<max_len):
            tokens.iloc[0] += [0] * (max_len - len(tokens.iloc[0]))
        tokens = pad_sequence(tokens.to_list(), batch_first=True)
        
        return tokens
        
    def create_tensor_for_paired_textdata(self, textdataA, textdataB, max_len=512, label_token_len=3):
        '''
        Convert the paired datasets of token ids to a single equal-width tensor(512 tokens) with zero padding on the right-hand-side
        Also create the segment id tensor (0 indicate tokens belong to dataset A, 1 indicate tokens belong to dataset B)
        '''
        max_text_len =  max_len - label_token_len
        half_max_text_len = int(max_text_len /2)
        segment_ids = []
        
        # Concat the token so that the comnbined length is < the max lenght for BERT input
        # And calculate segment id (0 indicating dataset A, 1 indicating dataset B)
        for row in range(len(textdataA["len"])):
            sentA_len = int(textdataA["len"].iloc[row])
            sentB_len = int(textdataB["len"].iloc[row])
            
            if sentA_len + sentB_len > max_text_len:
                if sentA_len < half_max_text_len:
                    textdataB["all_text_data"].iloc[row] = textdataB[
                        "all_text_data"].iloc[row][0:(max_text_len - sentA_len)]
                    
                elif sentB_len < half_max_text_len:
                    textdataA["all_text_data"].iloc[row] = textdataA[
                        "all_text_data"].iloc[row][:(max_text_len - sentB_len)]
                    
                    sentA_len = max_text_len - sentB_len
                    
                else:
                    textdataA["all_text_data"].iloc[row] = textdataA[
                        "all_text_data"].iloc[row][0:half_max_text_len]
                    textdataB["all_text_data"].iloc[row] = textdataB[
                        "all_text_data"].iloc[row][0:half_max_text_len]
                    
                    sentA_len = half_max_text_len
                    
            sentA_len += 1 # account for the [CLS] token
            segment_ids += [[0] * (sentA_len) + [1] * (max_text_len - sentA_len)]
        
        # Combine text data from dataset A and B in the form of
        # [CLS] [dataset A data] [SEP] [dataset B data] [SEP]
        textdataA["CLS"] = [[self.tokenizer.convert_tokens_to_ids("[CLS]")]] * len(textdataA)
        textdataA["SEP"] = [[self.tokenizer.convert_tokens_to_ids("[SEP]")]] * len(textdataA)
        
        textdataA = textdataA.reset_index()
        textdataB = textdataB.reset_index()
        combined_text_data = textdataA["CLS"] + textdataA["all_text_data"] + textdataA["SEP"] + textdataB["all_text_data"] + textdataA["SEP"] 
        
        # Add zero padding to the right hand size to create equal-length data
        if (len(combined_text_data.iloc[0])<max_len):
            combined_text_data.iloc[0] += [0] * (max_len - len(combined_text_data.iloc[0]))
            segment_ids[0] += [0] * (max_len - len(segment_ids[0]))    
            
        combined_text_data = combined_text_data.apply(lambda row: torch.tensor(row))
        combined_text_data = pad_sequence(combined_text_data.to_list(), batch_first=True)
        segment_ids = pad_sequence(list(map(torch.tensor, segment_ids)), batch_first=True)
        
        
        return combined_text_data, segment_ids

    def build_attention_mask(self, tensor, max_len=256):
        return tensor==0
