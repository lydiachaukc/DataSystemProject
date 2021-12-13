# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:48:10 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForSequenceClassification
from classification_NN import classification_NN
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CosineSimilarity

class NumBertMatcher_crossencoder(BertForSequenceClassification):
    """
    reference BertForTokenClassification class in the hugging face library
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, config, similarity_method = "cos"):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        if (config.num_input_dimension != 1):
            cos = CosineSimilarity()
            self.calculate_similiarity = lambda a, b: cos(a,b).view(-1,1)
            config.num_input_dimension = 1
        else:
            self.calculate_similiarity = self.calculate_difference
            
        self.classifier = classification_NN(
            inputs_dimension = config.num_input_dimension + config.text_input_dimension,
            num_hidden_lyr = config.num_hidden_lyr,
            dropout_prob = 0.2,
            bn = nn.BatchNorm1d(config.num_input_dimension)
            )
        
        self.norm_num_batch = nn.BatchNorm1d(config.num_input_dimension)
        self.init_weights()
        self.loss_fct  = CrossEntropyLoss()
        
        
    
    def forward(
            self,
            numerical_featuresA,
            numerical_featuresB,
            input_ids,
            attention_mask,
            labels,
            token_type_ids):
        
        # compute the cls embedding of the text features
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        cls_output = self.dropout(output[1])
        
        # calculate cossine similiary of numeric features
        numerical_features = self.calculate_similiarity(
            numerical_featuresA,
            numerical_featuresB)
        
        numerical_features = self.norm_num_batch(numerical_features)
        
        # Combined the text embedding with the similarity factor of numeric features
        all_features = torch.cat((cls_output, numerical_features), dim=1)
        
        # Calculate the logits, loss and accuracy
        logits = self.classifier(all_features)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.long().view(-1))
        else:
            loss=None
                
        return {'loss': loss,
                'logits': logits,
#                'accuracy': self.calculate_accuracy(labels, logits)
                }
    
    def calculate_difference(self, tensorA, tensorB):
        return torch.abs(tensorA - tensorB)
    
    # def calculate_accuracy(self, actual_labels, logits):
    #     _, predicted_labels = torch.max(logits, dim = 1)
        
    #     return (actual_labels == predicted_labels).sum().float() / len(actual_labels)