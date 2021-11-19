# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:05:20 2021

@author: lydia
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
                
class classification_NN(nn.Module):
    def __init__(self,
                 inputs_dimension,
                 num_hidden_lyr=2,
                 dropout_prob=0.5,
                 bn=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        
        output_dim = 2 # 0 for unmatch; 1 for match
        hidden_channels = [inputs_dimension for _ in range(num_hidden_lyr)]
        self.layer_channels = [inputs_dimension] + hidden_channels + [output_dim]
        
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                    for i in range(len(self.layer_channels) - 2)])))
        
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer)
        self.layers.append(final_layer)
        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]])
        
    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        return m
    
    def forward(self, data):
        """ forward propagate input
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [data]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        return layer_inputs[-1]
