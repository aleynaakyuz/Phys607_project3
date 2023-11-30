# This is a code for 1D mcmc

import numpy as np
from tqdm import tqdm
from testing import Gelman_Rubin


def mcmc(initial, data, post, prop, iterations):
    x = [initial]
    p = [post(data, x[-1])]
    for i in tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(data, x_test)
        acc = p_test - p[-1]
        u = np.log(np.random.uniform(0, 1))
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return np.array(x), np.array(p)
    

def mcmc_with_convergence(initial, data, post, prop, iterations, num_chains):
    chain_var1 = []
    chain_var2 = []
    
    for _ in range(num_chains):
            chain, prob = mcmc(initial, data, post, prop, iterations)
            var1 = chain[:,0]
            var2 = chain[:,1]
            chain_var1.append(var1)
            chain_var2.append(var2)
            
    R1 = Gelman_Rubin(chain_var1)    
    R2 = Gelman_Rubin(chain_var2)    
    
    return chain_var1, chain_var2, R1, R2