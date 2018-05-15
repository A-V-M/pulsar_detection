# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:13:25 2018

@author: andreas
"""
import numpy as np

def discretise_dist(data):
       
    max_val = int(np.ceil(np.max(data)))
        
    min_val = int(np.floor(np.min(data)))
        
    sample_size = len(data)
               
        #set number of bins to root of sample size
    n_bins = int(np.ceil(sample_size**(1./3)))
        
    bin_data = np.linspace(min_val,max_val,n_bins,retstep=True)
    
    bins = np.ceil(bin_data[0]).astype('int')

    bin_step = np.ceil(bin_data[1]).astype('int')
        
        #form discritised distributions
        
    histg = np.histogram(data, bins=bins, range=None, density=None)[0] 
            
        #sum-normalise    
        
    histg = histg.astype('float') / np.sum(histg)
    
    hist_data = dict({'hist': histg,'bins': bins, 'bin step': bin_step})

    return hist_data

def compute_KL(hist_data,setB):
    
    #A is set as the master distribution or the reference distribution
    #B is the theoretical model

    #remove zeros. KL does not deal well with zeros.
    
    histA = hist_data['hist']
    
    histB = np.histogram(setB, bins=hist_data['bins'], range=None, density=None)[0]
    
    histB = histB.astype('float') / np.sum(histB)

    histA[np.where(histA == 0)] = 1e-10
    
    histB[np.where(histB == 0)] = 1e-10
        
    KL = np.sum(histA * np.log(histA / histB), axis=0)
        
    return KL

def compute_sig_KL(KL,hist_data,sample_size,n_samples):

    synth_data=np.random.choice(hist_data['bins'][1:],size=(n_samples,sample_size),p=hist_data['hist'])
    synth_data=(synth_data - hist_data['bin step']) + (hist_data['bin step'] * np.random.rand(n_samples,sample_size))
    
    KLs = np.array([compute_KL(hist_data,x) for x in synth_data])
    
    sig = 1 - np.sum(KLs < KL) / float(n_samples)
                
    return sig,KLs
    
    
