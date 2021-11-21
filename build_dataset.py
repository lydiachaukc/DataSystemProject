# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 00:50:01 2021

@author: lydia
"""
import preprocess_text_data
import pandas as pd
import torch

class build_tensor_dataset:
    
    def __init__(self, preprocessed_obj, path):
        id_labs = pd.read_csv(path)
        id_A = id_labs.iloc[:,0]
        datasetA_numeric_data = preprocessed_obj.datasetA_numeric_data.iloc[id_A]
        datasetA_text_data = preprocessed_obj.datasetA_text_data.iloc[id_A]
        
        id_B = id_labs.iloc[:,1]
        datasetB_numeric_data = preprocessed_obj.datasetB_numeric_data.iloc[id_B]
        datasetB_text_data = preprocessed_obj.datasetB_text_data.iloc[id_B]
        
        self.combined_text_data, self.text_attention_mask, self.text_segment_ids = preprocessed_obj.text_preprocesser.build_datset(
            datasetA_text_data, datasetB_text_data)
        self.combined_numeric_data = self.combine_numeric_data(
            datasetA_numeric_data, datasetB_numeric_data)
        self.labels = torch.tensor(id_labs.iloc[:,2])
        
    def combine_numeric_data(self, datasetA, datasetB):
        datasetA = datasetA.reset_index(drop=True)
        datasetB = datasetB.reset_index(drop=True)
        return torch.tensor(pd.concat([datasetA, datasetB], axis=1).values)