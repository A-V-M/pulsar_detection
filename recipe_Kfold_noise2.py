# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 01:12:30 2018

@author: andreas
"""
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import scipy as sp
from random import randint, shuffle
from sklearn import linear_model, datasets
from scipy.stats import multivariate_normal
from sklearn import svm, tree
from sklearn.metrics import accuracy_score, recall_score,precision_score, make_scorer,fbeta_score


#from function_defs import *
#import pylab as plt
pulsars_ds = pd.DataFrame(np.random.rand(17898,9))
pulsars_ds[8] = (pulsars_ds[8] > 0.9).astype(float)

#import data
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

#set fold size [# non pulsar, # pulsars]

fold_size = [1620,162]

sample_size = [250,250]

k_folds = 10

n_estimators = np.arange(2,27)

max_depth = np.arange(1,11)

thresh = 0.85

F1s = np.zeros(k_folds)

precision_scores = np.zeros(k_folds)

recall_scores = np.zeros(k_folds)

meanF1s = np.zeros((len(n_estimators),len(max_depth)))

meanPrecision = np.zeros((len(n_estimators),len(max_depth)))

meanRecall = np.zeros((len(n_estimators),len(max_depth)))

test_times = np.zeros((len(n_estimators),len(max_depth)))

train_time = 0
test_time = 0

training_indices_kfolds,k_indices=gen_k_indices(nonpulsar_indices,pulsar_indices,np.max(n_estimators),sample_size,fold_size,k_folds)

estimators_indices = np.arange(0,np.max(n_estimators))

shuffle(estimators_indices)

for e,n_est in enumerate(n_estimators):
    
    est_subset = estimators_indices[np.arange(0,n_est)]
    
    for d,max_d in enumerate(max_depth):
                
        start_time = time.time()

        for k in np.arange(0,k_folds):
            
            training_indices = training_indices_kfolds[k,est_subset,:]
                            
            classifiers=DTrees_train(n_est,max_d,training_indices,dataset,actual)
                        
            start_time = time.time()                
            
            sum_preds,prob_preds=DTree_test(classifiers,dataset,k_indices[k,:],thresh)

            test_time += time.time() - start_time                       
            
            perf=assess_performance(sum_preds,prob_preds,k_indices[k,:],actual)
            
            F1s[k] = perf['F1']
            
            precision_scores[k] = perf['precision']
            
            recall_scores[k] = perf['recall']
                        
        test_times[e,d] = test_time / k_folds
        
        meanF1s[e,d] = np.mean(F1s)
    
        meanPrecision[e,d] = np.mean(precision_scores)
        
        meanRecall[e,d] = np.mean(recall_scores)
        
max_vals=np.unravel_index(np.argmax(meanF1s),np.shape(meanF1s))
print(meanF1s[max_vals])

print(['estimators', n_estimators[max_vals[0]]])

print(['max depth',max_depth[max_vals[1]]])


random_x=np.random.rand(1620,8) * np.mean(dataset,axis=0)
known_pulsars = dataset[pulsar_indices[0:162],:]
test_random=np.concatenate((random_x,known_pulsars))
targets = np.concatenate((np.zeros(1620),np.ones(162)))
np.sum(sum_preds * targets) / 50
np.sum(sum_preds * targets) / np.sum(sum_preds)
sum_preds,prob_preds=DTree_test(classifiers,test_random,np.arange(0,1782),thresh)


perf=assess_performance(sum_preds,prob_preds,k_indices[k,:],actual)


fig = plt.figure()
ax = fig.add_subplot(211)
cax =ax.matshow(meanF1s.T, interpolation='nearest') 
fig.colorbar(cax)
ax.set_xticklabels(['']+list(n_estimators[0:len(n_estimators):5]))
ax.set_yticklabels(['']+list(max_depth[0:10:2]))
plt.xlabel('# of base estimators')
plt.ylabel('max. tree depth')


ax = fig.add_subplot(212)
cax =ax.matshow(test_times.T, interpolation='nearest') 
fig.colorbar(cax)
ax.set_xticklabels(['']+list(n_estimators[0:len(n_estimators):5]))
ax.set_yticklabels(['']+list(max_depth[0:10:2]))
plt.xlabel('# of base estimators')
plt.ylabel('max. tree depth')


#output from 4/3/2018
#[10, 2, 0.94562499999999994]
#[10, 4, 0.96624999999999994]
#[10, 8, 0.95874999999999999]
#[10, 16, 0.96937499999999999]
#[10, 32, 0.96437499999999998]
#[10, 64, 0.96437499999999998]
#[20, 2, 0.94249999999999989]
#[20, 4, 0.96875]
#[20, 8, 0.96999999999999997]
#[20, 16, 0.97062500000000007]
#[20, 32, 0.97187499999999993]
#[20, 64, 0.97312500000000002]
#[30, 2, 0.948125]
#[30, 4, 0.97562499999999996]
#[30, 8, 0.96875]
#[30, 16, 0.96937499999999999]
#[30, 32, 0.97124999999999995]
#[30, 64, 0.97249999999999992]
#[40, 2, 0.94437499999999996]
#[40, 4, 0.97562499999999996]
#[40, 8, 0.97312500000000002]
#[40, 16, 0.97687499999999994]
#[40, 32, 0.97812499999999991]
#[40, 64, 0.97312500000000002]
#[50, 2, 0.94937499999999997]
#[50, 4, 0.96750000000000003]
#[50, 8, 0.97375]
#[50, 16, 0.97437499999999999]
#[50, 32, 0.97562499999999996]
#[50, 64, 0.979375]
#[60, 2, 0.95062499999999994]
#[60, 4, 0.9693750000000001]
#[60, 8, 0.97124999999999995]
#[60, 16, 0.97749999999999992]
#[60, 32, 0.97312500000000002]
#[60, 64, 0.97999999999999998]
#[70, 2, 0.953125]
#[70, 4, 0.97124999999999995]
#[70, 8, 0.97062499999999996]
#[70, 16, 0.97375]
#[70, 32, 0.97562499999999996]
#[70, 64, 0.96875]
#[80, 2, 0.94874999999999998]
#[80, 4, 0.97187500000000004]    
#[80, 8, 0.97937499999999988]
#[80, 16, 0.97750000000000004]
#[80, 32, 0.97687499999999994]
#[80, 64, 0.97625000000000006]
#[90, 2, 0.95250000000000001]
#[90, 4, 0.97312500000000002]
#[90, 8, 0.97687499999999994]
#[90, 16, 0.97937499999999988]
#[90, 32, 0.97312500000000002]
#[90, 64, 0.97312500000000002]
#[100, 2, 0.94562500000000005]
#[100, 4, 0.96999999999999997]
#[100, 8, 0.97312500000000002]
#[100, 16, 0.97875000000000001]