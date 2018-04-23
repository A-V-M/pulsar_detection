# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 19:03:50 2018

@author: andreas
"""
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

KLs = np.zeros(8)
sigs = np.zeros(8)

f, ax = plt.subplots(4,2)
ax = ax.reshape(1,8)[0]

  
for feat in range(0,8):
    
    setA = data[np.where(data[:,8]==0),feat].T
    setB = data[np.where(data[:,8]==1),feat].T
    
    #plt.subplot(4,2,feat+1)
    #plots normalised to have area under curve = 1
    ax[feat].hist(setA,100,normed=True)
    ax[feat].hist(setB,100,normed=True)
    ax[feat].set_title(feat+1)
            
    hist_data = discretise_dist(setA)
    KL=compute_KL(hist_data,setB)
    sig,KLsynth=compute_sig_KL(KL,hist_data,1638,100)
    
    KLs[feat] = KL

    sigs[feat] = sig

            

        
    
    










