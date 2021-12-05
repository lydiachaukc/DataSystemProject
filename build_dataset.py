# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 00:50:01 2021

@author: lydia
"""
import preprocess_data.preprocess_text_data
import pandas as pd
import torch

class build_tensor_dataset:
    def __init__(self, preprocessed_obj, path, is_cross_encoder = True):
        '''
        Given the testing set (with informaiton dataset A id, dataset B id, match/not match),
        Retrive the corresponding data from dataset A and B based on the ids
        and conver the data to tensor so that they are ready to be used in the mathcer models
    
        Outputs    
            For cross encoder:
            - combined_text_data: word tokens for the text data (dataset A and dataset B combined)
            - text_attention_mask: tensor of 0/1 indicating whether whether the token is a word token (1) or a dummy padding (0)
            - text_segment_ids: tensor of 0/1 indicating whether whether the token belongs to datasetA (0) or dataset B (1)
            - numeric_dataA: numberic feature data from dataset A
            - numeric_dataB: numberic feature data from dataset B
            - labels: 1d-tensour of 0/1 indicate that the rows from dataset A and dataset B refer to the same entity
            
            For cross encoder:
            - text_data_A: word tokens for the text data in dataset A
            - text_data_B: word tokens for the text data in dataset B
            - text_attention_mask_A: tensor of 0/1 indicating whether whether the token is a word token (1) or a dummy padding (0) in dataset A
            - text_attention_mask_B: tensor of 0/1 indicating whether whether the token is a word token (1) or a dummy padding (0) in dataset B
            - numeric_dataA: numberic feature data from dataset A
            - numeric_dataB: numberic feature data from dataset B
            - labels: 1d-tensor of 0/1 indicate that the rows from dataset A and dataset B refer to the same entity
        '''
        id_labs = pd.read_csv(path)
        id_A = id_labs.iloc[:,0]
        datasetA_numeric_data = preprocessed_obj.datasetA_numeric_data.iloc[id_A]
        datasetA_text_data = preprocessed_obj.datasetA_text_data.iloc[id_A]
        
        id_B = id_labs.iloc[:,1]
        datasetB_numeric_data = preprocessed_obj.datasetB_numeric_data.iloc[id_B]
        datasetB_text_data = preprocessed_obj.datasetB_text_data.iloc[id_B]
        
        if (is_cross_encoder):
            self.combined_text_data, self.text_attention_mask, self.text_segment_ids = preprocessed_obj.text_preprocesser.build_datset_for_cross_encoder(
                datasetA_text_data, datasetB_text_data)
        else:
            self.text_data_A, self.text_attention_maskA = preprocessed_obj.text_preprocesser.build_datset_for_bi_encoder(
                datasetA_text_data, segment_id=0)
            self.text_data_B, self.text_attention_maskB = preprocessed_obj.text_preprocesser.build_datset_for_bi_encoder(
                datasetA_text_data, segment_id=1)
            
        self.numeric_dataA = torch.tensor(datasetA_numeric_data.values, dtype=torch.float32)
        self.numeric_dataB = torch.tensor(datasetB_numeric_data.values, dtype=torch.float32)
        self.labels = torch.tensor(id_labs.iloc[:,2])