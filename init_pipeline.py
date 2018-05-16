# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:35:59 2018

@author: andreas
"""

from _multivariateGaussian import *
from _setupData import *
from _decisionTreesBagging import *
from _compareSets import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import time
import matplotlib.patches as mpatches


actual,nonpulsar_indices,pulsar_indices,dataset,pulsars_ds = prepData()


#%%

#############################
## Compare the two classes ##
#############################

print('Compare pulsars vs non-pulsars...')

KLs = np.zeros(8)
sigs = np.zeros(8)

f, ax = plt.subplots(4,2)
ax = ax.reshape(1,8)[0]

for feat in range(0,8):
    
    setA = dataset[np.where(actual==0),feat].T
    setB = dataset[np.where(actual==1),feat].T

    ax[feat].hist(setA,100,normed=True)
    ax[feat].hist(setB,100,normed=True)
    ax[feat].set_title(feat+1)
            
    hist_data = discretise_dist(setA)
    KL=compute_KL(hist_data,setB)
    sig,KLsynth=compute_sig_KL(KL,hist_data,1638,100)
    
    KLs[feat] = KL

    sigs[feat] = sig

    
skewness = dict({'pulsars': sp.stats.skew(pulsars_ds.values[actual==1,:-1]),
                 'non-pulsars': sp.stats.skew(pulsars_ds.values[actual==0,:-1])                                    
                })
                    
kurtosis = dict({'pulsars': sp.stats.kurtosis(pulsars_ds.values[actual==1,:-1]),
                 'non-pulsars': sp.stats.kurtosis(pulsars_ds.values[actual==0,:-1])                                   
                })


#%%
##########################
## Multivariate Gaussian##
##########################

print('Benchmark model [MV Gaussian]...')

fold_size = [1620,162]
k_folds = 10

thresh_vals = np.logspace(-8,2,250)

F1s = np.zeros(k_folds)
precisions = np.zeros(k_folds)
recalls = np.zeros(k_folds)
meanF1 = np.zeros(len(thresh_vals))
mean_precisions = np.zeros(len(thresh_vals))
mean_recalls = np.zeros(len(thresh_vals))
nPositives = np.zeros(len(thresh_vals))
nTPositive = np.zeros(len(thresh_vals))


training_indices_kfolds,k_indices = gen_k_indices_MV(actual,nonpulsar_indices,pulsar_indices,fold_size,k_folds)

test_time = 0

for thresh_i,thresh in enumerate(thresh_vals):
    
    for k in range(0,k_folds):
            
        training_data = dataset[training_indices_kfolds[k,:],:]
        test_data = dataset[k_indices[k,:],]
        test_target = actual[k_indices[k,:],]
        
        MV_dist = MV_Gaussian(training_data)
        
        start_time = time.time()                
    
        preds,pdfs,F1 = MV_Gaussian_test(MV_dist,test_data,test_target,thresh)
        
        test_time += time.time() - start_time                       
    
        perf=assess_performance(preds,MV_dist.pdf(test_data),k_indices[k,:],actual)
        
        F1s[k]=F1
        precisions[k] = perf['precision']
        recalls[k] = perf['recall']
        
    avg_test_time = test_time / k_folds
    meanF1[thresh_i] = np.mean(F1s)
    mean_precisions[thresh_i] = np.mean(precisions)
    mean_recalls[thresh_i] = np.mean(recalls)
    
    nPositives[thresh_i] = np.dot(preds,preds.astype('float'))
    nTPositive[thresh_i] = np.dot(preds,test_target)

max_i=np.unravel_index(np.argmax(meanF1),np.shape(meanF1))

   
#    print(avg_test_time)
#    print([['F1',np.mean(F1s)],['Precision',np.mean(precisions)],['Recall',np.mean(recalls)]])
print(meanF1[max_i],'at optimum threshold:',thresh_vals[max_i])

plt.plot(thresh_vals,nPositives,'b')
plt.plot(thresh_vals,nTPositive,'r')
plt.plot(thresh_vals,[162]*250,'k--')

red_patch = mpatches.Patch(color='red', label='True positive')
blue_patch = mpatches.Patch(color='b', label='Total positive')
plt.legend(handles=[red_patch,blue_patch],loc=2)
plt.xlabel('threshold')
plt.ylabel('# of instances')

print('Testing on random data...')
random_x=np.random.rand(1620,8) * np.mean(dataset,axis=0)
known_pulsars = dataset[pulsar_indices[0:162],:]
test_random=np.concatenate((random_x,known_pulsars))
targets = np.concatenate((np.zeros(1620),np.ones(162)))
preds,pdfs,F1=MV_Gaussian_test(MV_dist,test_random,1-targets,thresh_vals[max_i])
perf_random=assess_performance(preds,MV_dist.pdf(test_random),np.arange(0,1782),targets)

print(perf_random)

#%%
###############################
## Decision Trees w/ Bagging ##
###############################

print('Running Decision Trees & Bagging...')

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

training_indices_kfolds,k_indices=gen_k_indices(actual,nonpulsar_indices,pulsar_indices,np.max(n_estimators),sample_size,fold_size,k_folds)

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
np.sum(sum_preds * targets) / np.sum(sum_preds)
sum_preds,prob_preds=DTree_test(classifiers,test_random,np.arange(0,1782),thresh)

perf_random=assess_performance(sum_preds,prob_preds,k_indices[k,:],actual)
print(perf_random)


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


