import numpy as np
import pandas as pd

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

    #split by year wise
    def split_by_year_wise(self):
        date_series = self.df[self.date]
        
        #sort by dataset by index
        self.df.index = date_series
        self.df = self.df.sort_index()
        
        #get the year column
        self.df['year'] = pd.DatetimeIndex(self.df[self.date]).year

        #get year data
        year = self.df['year'].value_counts().keys().values.tolist()
        year.sort()
        train_set = year[:-2]
        test_set = year[-2:] #get last 2 years
        
        train_split = self.df[self.df['year'].isin(train_set)][[self.forecastcolumn]]
        test_split = self.df[self.df['year'].isin(test_set)][[self.forecastcolumn]]
        
        return train_split, test_split