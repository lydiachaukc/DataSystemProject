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
    """
    def __init__(self,task_config, lm):
        self.config = task_config
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])

    def tokenize_dataset(self, dataset, max_len=256):
        tokenized_dataset =  pd.DataFrame(np.zeros((len(dataset),1)), columns=["len"])
        tokenized_dataset["all_text_data"] = np.empty((len(dataset), 0)).tolist()
        
        for col in dataset:
            tokenized_dataset[col] = None
            print("col:", col)
            tokenized_dataset["all_text_data"] += dataset[col].apply(
                lambda elem: self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(self.tokenize_field(elem))))
        tokenized_dataset["len"] = tokenized_dataset["all_text_data"].apply(lambda elem: len(elem))
        
        return tokenized_dataset[["all_text_data","len"]]
        
    def tokenize_field(self, field):
        tokens = ""
        words = field.split(' ')
        for word in words:
            if word not in stopwords:
                tokens += word + " [SEP] "
        return tokens
    
    def build_datset(self, textdataA, textdataB):
        tokens, segment_ids = self.combine_textdataA_textdataB(textdataA, textdataB)
        attention_mask = self.build_attention_mask(tokens)
        return tokens, attention_mask, segment_ids
            
    def combine_textdataA_textdataB(self, textdataA, textdataB, max_len=512, label_token_len=3):
        max_text_len =  max_len - label_token_len
        half_max_text_len = int(max_text_len /2)
        segment_ids = []
        
        for row in range(len(textdataA["len"])):
            sentA_len = int(textdataA["len"].iloc[row])
            sentB_len = int(textdataB["len"].iloc[row])
            
            if sentA_len + sentB_len > max_text_len:
                if sentA_len < half_max_text_len:
                    textdataB["all_text_data"].iloc[row] = textdataB[
                        "all_text_data"].iloc[row][0:(max_text_len - sentA_len)]
                    
                    sentB_len = max_text_len - sentA_len
                    
                elif sentB_len < half_max_text_len:
                    textdataA["all_text_data"].iloc[row] = textdataA[
                        "all_text_data"].iloc[row][:(max_text_len - sentB_len)]
                    
                    sentA_len = max_text_len - sentB_len
                    
                else:
                    textdataA["all_text_data"].iloc[row] = textdataA[
                        "all_text_data"].iloc[row][0:half_max_text_len]
                    textdataB["all_text_data"].iloc[row] = textdataB[
                        "all_text_data"].iloc[row][0:half_max_text_len]
                    
                    sentA_len, sentB_len = half_max_text_len, half_max_text_len
            segment_ids += [[0] * (sentA_len + 1) + [1] * sentB_len]
                    
        textdataA["CLS"] = [[self.tokenizer.convert_tokens_to_ids("[CLS]")]] * len(textdataA)
        textdataA["SEP"] = [[self.tokenizer.convert_tokens_to_ids("[SEP]")]] * len(textdataA)
        
        textdataA = textdataA.reset_index()
        textdataB = textdataB.reset_index()
        combined_text_data = textdataA["CLS"] + textdataA["all_text_data"] + textdataA["SEP"] + textdataB["all_text_data"] + textdataA["SEP"] 
        
        if (len(combined_text_data.iloc[0])<max_len):
            combined_text_data.iloc[0] += [0] * (max_len - len(combined_text_data.iloc[0]))
            segment_ids[0] += [0] * (max_len - len(segment_ids[0]))
        combined_text_data = combined_text_data.apply(lambda row: torch.tensor(row))
        
        combined_text_data = pad_sequence(combined_text_data.to_list(), batch_first=True)
        segment_ids = pad_sequence(list(map(torch.tensor, segment_ids)), batch_first=True)
        
        return combined_text_data, segment_ids

    def build_attention_mask(self, tensor, max_len=256):
        return tensor==0
