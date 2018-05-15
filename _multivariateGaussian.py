# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:12:45 2018

@author: andreas
"""
from sklearn.metrics import f1_score,roc_auc_score
import numpy as np
from random import randint, shuffle
from scipy.stats import multivariate_normal



def assess_performance(sum_preds,prob_preds,test_indices,actual):
    
    batch_size = len(test_indices)
    
    #precision
    
    precision = np.sum(actual[test_indices] * sum_preds) / np.sum(sum_preds)
    
    #accuracy
    
    accuracy = np.sum((actual[test_indices] - sum_preds)==0) / float(batch_size)

    #recall
    
    recall = np.sum(actual[test_indices] * sum_preds) / np.sum(actual[test_indices])
    
    #F1

    F1 = f1_score(actual[test_indices].T,sum_preds.T)    
    
    #AUC
    
    AUC = roc_auc_score(actual[test_indices].T,prob_preds.T)
    
    #information gain
    
    pos_rate = np.sum(sum_preds) / batch_size
    
    neg_rate = np.sum(1-sum_preds) / batch_size
    
    NPV = np.sum((1-actual[test_indices]) * (sum_preds)) / np.sum(sum_preds) #negative predictive value
    
    PartI = pos_rate * ((precision * np.log(precision)) + (NPV * np.log(NPV)))

    PartII = neg_rate * ((precision * np.log(precision)) + (NPV * np.log(NPV)))

    IG = -(PartI + PartII)
    
    return dict({'AUC':round(np.mean(AUC),2),
                 'F1':round(np.mean(F1),2),
                 'precision': round(np.mean(precision),2),
                 'NPV': round(np.mean(NPV),2),
                 'accuracy': round(np.mean(accuracy),2),
                 'recall': round(np.mean(recall),2),
                 'IG': round(np.mean(IG),2)
                 })


def gen_k_indices_MV(actual,nonpulsar_indices,pulsar_indices,fold_size,k_folds):

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

    
    
    
    