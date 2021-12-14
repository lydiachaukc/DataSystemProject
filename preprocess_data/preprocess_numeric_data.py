# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:21:24 2021

@author: lydia
"""

class Preprocess_numeric_data:
    """To handle dirty data in numeric columns and to normalize the numeric data
    """
    def __init__(self, dataset, store_preprocess_data = False):
        self.dataset = dataset
        self.handle_dirty_numeric_data()
        
    def handle_dirty_numeric_data(self):
        for name in self.dataset.columns:
            column = self.dataset[name]
            
            # # Normalize the data
            self.dataset[name] = (column-column.mean())/column.std()
            self.dataset[name] = self.dataset[name].fillna(0)
            
            
            # Fill missing data with column mean
            self.dataset[name] = self.dataset[name].fillna(column.mean())