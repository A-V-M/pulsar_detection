# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:33:24 2018

@author: andreas
"""

from sklearn.metrics import f1_score,roc_auc_score
import numpy as np
from random import randint, shuffle
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier



def gen_k_indices(actual,nonpulsar_indices,pulsar_indices,n_estimators,sample_size,fold_size,k_folds):
    
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
        
    training_indices_kfolds = np.zeros((k_folds,n_estimators,sum(sample_size))).astype('int')

    for k in k_range:
        
        training_indices = np.ndarray.flatten(k_indices[k_range != k,:])
            
        nonpulsar_training_indices = training_indices[actual[training_indices]==0]
        pulsar_training_indices = training_indices[actual[training_indices]==1]
                 
        for n in range(0,n_estimators):
                
            shuffle(nonpulsar_training_indices)
            shuffle(pulsar_training_indices)   
        
            inds = np.concatenate((nonpulsar_training_indices[0:sample_size[0]],pulsar_training_indices[0:sample_size[1]]))
            shuffle(inds)
            training_indices_kfolds[k,n,:] = inds
        
    return training_indices_kfolds,k_indices

def gen_indices(test_size,sample_size,n_estimators,n_test_batches,nonpulsar_indices,pulsar_indices):
    
    shuffle(nonpulsar_indices)
    shuffle(pulsar_indices)
    
    nonpulsar_training_indices = nonpulsar_indices[:len(nonpulsar_indices)-(test_size[0])]
    pulsar_training_indices = pulsar_indices[:len(pulsar_indices)-(test_size[1])]
    
    nonpulsar_test_indices = nonpulsar_indices[-(test_size[0]):]
    pulsar_test_indices = pulsar_indices[-(test_size[1]):]
    
    training_indices = np.zeros((n_estimators,sum(sample_size)))

    
    for n in range(0,n_estimators):
        
        shuffle(nonpulsar_training_indices)
        shuffle(pulsar_training_indices)   

        i = np.concatenate((nonpulsar_training_indices[0:sample_size[0]],pulsar_training_indices[0:sample_size[1]]))
        training_indices[n,:] = i

    factor=10.
    nonpulsar = int(np.ceil((np.array(test_size)/factor)[0]))
    pulsar = int(np.ceil((np.array(test_size)/factor)[1]))
    
    batch_size = nonpulsar+pulsar
    
    test_indices = np.zeros((n_test_batches,batch_size))
    
    for m in range(0,n_test_batches):
        
        shuffle(nonpulsar_test_indices)
        shuffle(pulsar_test_indices) 
        
        master_test_indices = np.concatenate((nonpulsar_test_indices,pulsar_test_indices))                  
        
        shuffle(master_test_indices)
        
        test_indices[m,:] = master_test_indices[0:batch_size]
        
                    
    return training_indices.astype('int'), test_indices.astype('int')
        

          
## Decision Tree Classifier
        
def DTrees_train(n_estimators,max_depth,training_indices,dataset,actual):

    estimators = range(n_estimators)
    
    classifiers = [0] * n_estimators
    
    for n in estimators:
        
        indxs = training_indices[n,:]
    
        training_set = dataset[indxs,:]
    
        clf = DecisionTreeClassifier(random_state=0,max_depth=max_depth)
        #clf=svm.SVC(random_state=0, C=1,kernel="linear")
        clf.fit(training_set,actual[indxs])
        classifiers[n] = np.array(clf)
        
    return classifiers
    
                 
def DTree_test(classifiers,dataset,test_indices,thresh):
    
    n_estimators = len(classifiers)
    estimators = range(0,n_estimators)
    
    batch_size = len(test_indices.T)
    
    n_test_batches = len(test_indices)
    
    preds = np.zeros((n_estimators,batch_size))
    
    for n in estimators:
            
        preds[n,:] = classifiers[n].item().predict(dataset[test_indices,:])
                   
    sum_preds = ((np.sum(preds,axis=0) / n_estimators) > thresh)
    prob_preds = np.sum(preds,axis=0) / float(n_estimators)
       
    return sum_preds,prob_preds


