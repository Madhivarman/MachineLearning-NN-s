from statsmodels.tsa.stattools import adfuller
import pandas as pd

class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None
    
    def ADF_Stationarity_Test(self, item_no, timeseries, printResults = True):
        #Dickey-Fuller test:
        #autolag - Akaike Information Criterion
        adfTest = adfuller(timeseries, autolag='AIC')
        
        self.pValue = adfTest[1]
        
        if (self.pValue<self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False
        
        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
            stats = {} #stats dump
            stats['Item'] = item_no
            stats['ADF Test Statistics'] = adfTest[0]
            stats['P-Value'] = adfTest[1]
            stats['Lags Used'] = adfTest[2]
            stats['Observed Used'] = adfTest[3]
            stats['isStationary'] = self.isStationary
            
            stats['critical_values'] = {}
            #Add Critical Values
            for key,value in adfTest[4].items():
                stats['critical_values'][key] = value
                dfResults['Critical Value (%s)'%key] = value
            
            
            return stats