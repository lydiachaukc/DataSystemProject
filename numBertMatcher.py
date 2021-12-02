# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:48:10 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

#from transformers import AutoModel
from transformers import BertForSequenceClassification
from classification_NN import classification_NN
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CosineSimilarity

class NumBertMatcher(BertForSequenceClassification):
    """
    reference BertForTokenClassification class in the hugging face library
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = classification_NN(
            inputs_dimension = config.num_input_dimension + config.text_input_dimension,
            num_hidden_lyr = config.num_hidden_lyr,
            dropout_prob = 0.8,
            bn = nn.BatchNorm1d(config.num_input_dimension).double()
            )
        self.norm_num_batch = nn.BatchNorm1d(config.num_input_dimension).double()
        self.init_weights()
        self.cos = CosineSimilarity()
              
    def forward(
            self,
            numerical_featuresA,
            numerical_featuresB,
            input_ids,
            attention_mask,
            labels,
            token_type_ids):
        
        #Run the text through the BERT model
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        cls_output = self.dropout(output[1])
        
        # calculate cossine similiary of numeric data
        numerical_features = self.cos(
            numerical_featuresA,
            numerical_featuresB)
        
        numerical_features = self.norm_num_batch(numerical_features)
        
        #concat everything to a vector
        all_features = torch.cat((cls_output, numerical_features), dim=1)
    
        logits = self.classifier(all_features)
        
        if labels is not None:
            loss_fct  = CrossEntropyLoss()
            labels = labels.long()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss=None
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
