import numpy as np

def Gelman_Rubin(var_ch):
  '''function that tests convergence of MCMC and returns a R-value. R-values close to 1 indicates good convergence'''  
    J = len(var_ch) #Number of chains
    L = var_ch[0].shape[0] #length of each chain
    
    var_mean = []
    for i in range(J):
        var_mean.append(np.mean(var_ch[i])) 
        
    chain_mean = np.array(var_mean) #the mean for each chain    
    grand_mean = np.mean(chain_mean) #grand mean across chains    
    B = L/(J-1)* np.sum((chain_mean - grand_mean)**2) #between chain variance
    
    var_variance = []
    for i in range(J):
        var_variance.append(np.var(var_ch[i])) 
        
    s2j = np.array(var_variance) #within chain variance for each chain
    W = np.mean(s2j) #average within-chain variance
    R = ((L-1)/L * W + B/L)/ W 
    return R
