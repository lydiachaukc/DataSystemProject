# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:35:04 2021

@author: lydia
"""
import pandas as pd
import numpy as np

from preprocess_text_data import preprocess_text_data

class Load_and_preprocess:
    def __init__(self, task_config, lm,
                 data_was_preprocessed = False, store_preprocessed_data = False):
        self.task_config = task_config
        datasetA = pd.read_csv(task_config["tableA"], index_col = "id")
        datasetB = pd.read_csv(task_config["tableB"], index_col = "id")
        
        datasetA["Price"] = pd.to_numeric(datasetA["Price"], errors='coerce')
        datasetB["Price"] = pd.to_numeric(datasetB["Price"], errors='coerce')
        
        self.numeric_columns_names = {"datasetA": [], "datasetB": []}
        self.text_columns_names = {"datasetA": [], "datasetB": []}
        
        self.identify_columns_types("datasetA", datasetA)
        self.identify_columns_types("datasetB", datasetB)
        self.text_preprocesser = preprocess_text_data(task_config, lm)
        
        if data_was_preprocessed:
            self.read_preprocessed_data()
            
        else:
            self.datasetA_numeric_data = Preprocess_numeric_data(
                datasetA[self.numeric_columns_names["datasetA"]]).dataset
            self.datasetB_numeric_data = Preprocess_numeric_data(
                datasetB[self.numeric_columns_names["datasetB"]]).dataset
                    
            self.datasetA_text_data = self.text_preprocesser.tokenize_dataset(
                datasetA[self.text_columns_names["datasetA"]])
            self.datasetB_text_data = self.text_preprocesser.tokenize_dataset(
                datasetB[self.text_columns_names["datasetB"]])
            
            if store_preprocessed_data:
                self.export_preprocessed_data()
        
    def identify_columns_types(self, dataset_name, data_frame):
        """" Populate numeric_columns_names and text_columns_names with
        the corresponding column names based on the elements in the dataset
        """
        for col_name in data_frame.columns[1:]:
            if np.issubdtype(data_frame[col_name].dtype, np.number):
                self.numeric_columns_names[dataset_name].append(col_name)
            else:
                self.text_columns_names[dataset_name].append(col_name)
                data_frame[col_name] = data_frame[col_name].astype(str)
    
    def export_preprocessed_data(self):
        folder_director = self.task_config[
            "tableA"][0:(self.task_config["tableA"].rfind("/")+1)]
        self.datasetA_numeric_data.to_csv(folder_director + "datasetA_numeric_data.csv")
        self.datasetB_numeric_data.to_csv(folder_director + "datasetB_numeric_data.csv")
        
        self.datasetA_text_data.to_csv(folder_director + "datasetA_text_data.csv")
        self.datasetB_text_data.to_csv(folder_director + "datasetB_text_data.csv")
        
    def read_preprocessed_data(self):
        folder_director = self.task_config[
            "tableA"][0:(self.task_config["tableA"].rfind("/")+1)]
        
        self.datasetA_numeric_data = pd.read_csv(
            folder_director + "datasetA_numeric_data.csv", index_col = "id")
        self.datasetB_numeric_data = pd.read_csv(
            folder_director + "datasetB_numeric_data.csv", index_col = "id")
        
        self.datasetA_text_data = pd.read_csv(
            folder_director + "datasetA_text_data.csv", index_col = 0,
            converters={"all_text_data": lambda x: list(map(int, x.strip("[]").split(", ")))})
        self.datasetB_text_data = pd.read_csv(
            folder_director + "datasetB_text_data.csv", index_col = 0,
            converters={"all_text_data": lambda x: list(map(int, x.strip("[]").split(", ")))})

            
        
        
        
class Preprocess_numeric_data:
    """To handle dirty data in numeric columns and to normalize the numeric data
    """
    def __init__(self, dataset, store_preprocess_data = False):
        self.dataset = dataset
        self.handle_dirty_numeric_data_and_normalize()
        
    def handle_dirty_numeric_data_and_normalize(self):
        for name in self.dataset.columns:
            column = self.dataset[name]
            # method 1: null all non-numeric data
            #column[(not column.str.isnumeric())] = None
            
            # method 2: try to guess the number
            
            # normalize the data
            self.dataset[name] = (column-column.mean())/column.std()
            self.dataset[name] = self.dataset[name].fillna(0)

