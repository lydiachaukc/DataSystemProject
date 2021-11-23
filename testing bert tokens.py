# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:58:49 2021

@author: lydia
"""
import numpy as np
import pandas as pd

lower_bound = 0
upper_bound = 100
n = 1000
numbers = np.linspace(lower_bound, upper_bound, n)


from transformers import AutoTokenizer
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

lm ='bert'
tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
tokens = tokenizer.convert_tokens_to_ids(pd.DataFrame(numbers)[0].astype(str))