# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:12:45 2018

@author: andreas
"""

import pandas as pd
import numpy as np
import scipy as sp
from random import randint, shuffle
from sklearn import linear_model, datasets
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score,f1_score

pulsars_ds = pd.read_csv('HTRU_2.csv',header=None)


#get labels and category indices
actual = pulsars_ds[8].values
nonpulsar_indices = pulsars_ds[pulsars_ds[8]==0].index.values
pulsar_indices = pulsars_ds[pulsars_ds[8]==1].index.values  
                           
                           

#cubic root plus addition of constant to remove negative values
pulsars_trans = (pulsars_ds[pulsars_ds.columns[0:8]]+4)**(1./3)
#pulsars_trans = (pulsars_ds[pulsars_ds.columns[0:8]])

nonpulsar_transformed = pulsars_trans[pulsars_ds[8]==0]
pulsar_transformed = pulsars_trans[pulsars_ds[8]==1]

dataset = pulsars_trans.values
     
fold_size = [1620,162]
thresh = 0.001
k_folds = 10

def gen_k_indices_MV(nonpulsar_indices,pulsar_indices,fold_size,k_folds):

    shuffle(nonpulsar_indices)
    shuffle(pulsar_indices)
    
    np_ranges=range(0,len(nonpulsar_indices),fold_size[0])   
    p_ranges=range(0,len(pulsar_indices),fold_size[1])
        
    k_indices = np.zeros((k_folds,np.sum(fold_size))).astype('int')
    
    k_range = np.arange(0,k_folds)
    
    for k in k_range:
        
        np_batch = nonpulsar_indices[np_ranges[k]:np_ranges[k+1]]
        
        p_batch = pulsar_indices[p_ranges[k]:p_ranges[k+1]]
        
        k_indices[k,:] = np.concatenate((np_batch,p_batch))
    
    training_set_size = (k_folds-1) * (np.sum(fold_size)) - ((k_folds-1) * fold_size[1])
    
    training_indices_kfolds = np.zeros((k_folds,training_set_size)).astype('int')

    for k in k_range:
        
        training_indices = np.ndarray.flatten(k_indices[k_range != k,:])
        
        training_indices = training_indices[actual[training_indices]==0]
        
        training_indices_kfolds[k,:] = training_indices
                               
    return training_indices_kfolds, k_indices



def MV_Gaussian(data):
    
    cov=np.cov(data.T)

    means=np.mean(data,axis=0)

    MV_dist = multivariate_normal(mean=means,cov=cov)
    
    return MV_dist

def MV_Gaussian_test(MV_dist,test_data,actual,thresh):
    
    preds = MV_dist.pdf(test_data) < thresh
    pdfs = MV_dist.pdf(test_data)
    
    AUC = roc_auc_score(actual,pdfs)
    
    F1 = f1_score(actual,preds)
    
    return preds, pdfs, F1

F1s = np.zeros(k_folds)
precisions = np.zeros(k_folds)
recalls = np.zeros(k_folds)
IGs = np.zeros(k_folds)
NPVs = np.zeros(k_folds)
training_indices_kfolds,k_indices = gen_k_indices_MV(nonpulsar_indices,pulsar_indices,fold_size,k_folds)

test_time = 0


for k in range(0,k_folds):
        
    training_data = dataset[training_indices_kfolds[k,:],:]
    test_data = dataset[k_indices[k,:],]
    test_target = actual[k_indices[k,:],]
    
    MV_dist = MV_Gaussian(training_data)
    
    start_time = time.time()                

    preds,pdfs,F1 = MV_Gaussian_test(MV_dist,test_data,test_target,thresh)
    
    test_time += time.time() - start_time                       

    perf=assess_performance(preds,MV_dist.pdf(test_data),k_indices[k,:],actual)
    
    F1s[k]=perf['F1']
    precisions[k] = perf['precision']
    recalls[k] = perf['recall']
    IGs[k] = perf['IG']
    NPVs[k] = perf['NPV']
    
avg_test_time = test_time / k_folds

print(avg_test_time)
    
print(np.mean(F1s))
print(np.mean(precisions))
print(np.mean(recalls))
print(np.mean(IGs))
print(np.mean(NPVs))

#random_x=np.random.rand(1620,8) * np.mean(dataset,axis=0)
#known_pulsars = dataset[pulsar_indices[0:162],:]
#test_random=np.concatenate((random_x,known_pulsars))
#targets = np.concatenate((np.zeros(1620),np.ones(162)))
#preds,pdfs,F1=MV_Gaussian_test(MV_dist,test_random,1-targets,thresh)
#perf_random=assess_performance(preds,MV_dist.pdf(test_random),np.arange(0,1782),targets)

#preds,pdfs,AUCs=MV_Gaussian_test(MV_dist,test_random,1-targets,thresh)

    
    
    
    