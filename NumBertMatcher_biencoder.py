# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:39:40 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForSequenceClassification
from classification_NN import classification_NN
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CosineSimilarity, BCELoss

class NumBertMatcher_biencoder(BertForSequenceClassification):
    """
    reference BertForTokenClassification class in the hugging face library
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.classifier = classification_NN(
        #     inputs_dimension = config.num_input_dimension + config.text_input_dimension,
        #     num_hidden_lyr = config.num_hidden_lyr,
        #     dropout_prob = 0.8,
        #     bn = nn.BatchNorm1d(config.num_input_dimension)
        #     )
        self.norm_num_batch = nn.BatchNorm1d(config.num_input_dimension)
        self.init_weights()
        
        self.calculate_similiarity = CosineSimilarity()
        self.loss_fct  = BCELoss()
    
    def forward(
            self,
            numerical_featuresA,
            numerical_featuresB,
            input_idsA,
            attention_maskA,
            input_idsB,
            attention_maskB,
            labels):
        
        # Compute the cls embedding of the text features
        textoutputA = self.bert(
            input_ids = input_idsA,
            attention_mask = attention_maskA
            )
        cls_output_A = self.dropout(textoutputA[1])
                
        textoutputB = self.bert(
            input_ids = input_idsB,
            attention_mask = attention_maskB
            )
        cls_output_B = self.dropout(textoutputB[1])
        
        # Combine all the text embeddings with numeric features
        combined_features_A = torch.cat((cls_output_A, numerical_featuresA), dim=1)
        combined_features_B = torch.cat((cls_output_B, numerical_featuresB), dim=1)
        
        # Calculate similiary of paired data
        similiarity = self.calculate_similiarity(
            combined_features_A,
            combined_features_B).view(-1,1)
        
        # Calculate the logits and loss
        if labels is not None:
            loss = self.loss_fct(similiarity, labels.float().view(-1,1))
        else:
            loss=None
        
        return {'loss': loss,
                'similarity': similiarity
                }
    