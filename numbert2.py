# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:48:10 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModel
from transformers import BertForSequenceClassification
from classification_NN import classification_NN
from transformers.modeling_outputs import TokenClassifierOutput

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}
class NumBert(BertForSequenceClassification):
    """
    reference BertForTokenClassification class in the hugging face library
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForTokenClassification
    """
    def __init__(self, config):
        super().__init__(config)
        
        #self.bert = AutoModel.from_pretrained(lm_mp[config.lm]).double()
        self.dropout = nn.Dropout(0.5)
        self.classifier = classification_NN(
            inputs_dimension = config.num_input_dimension + config.text_input_dimension,
            num_hidden_lyr = config.num_hidden_lyr,
            dropout_prob = 0.5,
            bn = nn.BatchNorm1d(config.num_input_dimension).double()
            )
        self.norm_num_batch = nn.BatchNorm1d(config.num_input_dimension).double()
        self.init_weights()
              
    def forward(
            self,
            numerical_features,
            input_ids,
            attention_mask,
            labels):
        
        #Run the text through the BERT model
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        
        cls_output = self.dropout(output[1])
        
        # batch normalization
        numerical_features = self.norm_num_batch(numerical_features)
        
        #concat everything to a vector
        all_features = torch.cat((cls_output, numerical_features), dim=1)
        
        # Output Classifier / Regression
        logits = self.classifier(all_features)
                    
        # Loss calculaiton
        if labels is not None:
            calculate_loss = CrossEntropyLoss(weight = None)
            labels = labels.long()
            loss = calculate_loss(logits.view(-1,2),labels.view(-1))
        else:
            loss=None
        
        return TokenClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=output.hidden_states,
                    attentions=output.attentions)
    

