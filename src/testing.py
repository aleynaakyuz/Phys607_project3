import numpy as np

def Gelman_Rubin(var_ch):
    
    J = len(var_ch)
    L = var_ch[0].shape[0]
    
    var_mean = []
    for i in range(J):
        var_mean.append(np.mean(var_ch[i]))
        
    chain_mean = np.array(var_mean)    
    grand_mean = np.mean(chain_mean)     
    B = L/(J-1)* np.sum((chain_mean - grand_mean)**2)
    
    var_variance = []
    for i in range(J):
        var_variance.append(np.var(var_ch[i]))
        
    s2j = np.array(var_variance)
    W = np.mean(s2j)
    R = ((L-1)/L * W + B/L)/ W
    return R
