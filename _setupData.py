# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:31:06 2018

@author: andreas
"""
import pandas as pd

def prepData():
       
    pulsars_ds = pd.read_csv('HTRU_2.csv',header=None)
      
    #get labels and category indices
    actual = pulsars_ds[8].values
    nonpulsar_indices = pulsars_ds[pulsars_ds[8]==0].index.values
    pulsar_indices = pulsars_ds[pulsars_ds[8]==1].index.values  
                                 
    #cubic root plus addition of constant to remove negative values
    dataset = ((pulsars_ds[pulsars_ds.columns[0:8]]+4)**(1./3)).values
    
    return actual,nonpulsar_indices,pulsar_indices,dataset,pulsars_ds
    
    
    
    