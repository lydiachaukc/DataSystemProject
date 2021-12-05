# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:58:04 2021

@author: lydia
"""
import pandas as pd
import torch
import datetime as datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
        
def setup_cuda():
  if torch.cuda.is_available():    
      print('Running on GPU')
      return torch.device("cuda") 
  else:
      print('Running on CPU')
      return torch.device("cpu")

def add_record(dataframe, time = "", model = "", epoch = "", batch = "", loss = "", purpose = "", data = ""):
    return dataframe.append(
        {"Time": time, "Model": model, "Epochs": epoch, "Batch": batch, "Loss": loss, "Purpose": purpose, "Data":data},
        ignore_index=True)