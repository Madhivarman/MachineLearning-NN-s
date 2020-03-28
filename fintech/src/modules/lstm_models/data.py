import numpy as np

class Dataset:
    
    #initial class objects
    def __init__(self, dataset, datecolumn, forecastcolumn):
        self.df = dataset.copy()
        self.date = datecolumn
        self.forecastcolumn = forecastcolumn
        
    #normal split
    def normal_split(self):
        date_series = self.df[self.date]
        #sort the dataset by index
        self.df.index = date_series
        self.df = self.df.sort_index()
        
        #take 70% data for train, 30% for evaluation dataset
        train_size = int(self.df.shape[0] * 0.77)
        #split the dataset
        train_split, eval_split = self.df.iloc[0:train_size], self.df.iloc[train_size:]
        #train values
        train_split, eval_split = train_split[[self.forecastcolumn]], eval_split[[self.forecastcolumn]]
        
        return train_split, eval_split