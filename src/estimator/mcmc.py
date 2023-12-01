# This is a code for 2D mcmc

import numpy as np
from tqdm import tqdm
from testing import Gelman_Rubin



def mcmc(initial, data, inp, post, prop, iterations):
    """ 
   Function performing the 2D MCMC sampling with 5 
   parameters which illustrates (a) initial guess for 
   the estimated parameter, (b) observed data, 
   (c) posterior probability function,
   (d) proposal function, and (e) number oof iterations

   Parameters:
   ------------
   initial : numpy array, initial conditions of parameters.
   data : numpy array, data.
   post: function, posterior function.
   prop : function, proposal function.
   iterations : int, number of itterations that mcmc alghoritm runs.

   Returns:
   --------
   x_arr : numpy array, chain 
   p_arr : numpt array, posterior prpbabilities
   
   """

    x = [initial]
    p = [post(data, inp, x[-1])]
    for i in tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(data, inp, x_test)
        acc = p_test - p[-1]
        u = np.log(np.random.uniform(0, 1))
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
        x_arr = np.array(x)
        p_arr = np.array(p)
    return x_arr, p_arr  #returns array of sampled parameters, and array of corresponding posterior probabilities
    

def mcmc_with_convergence(initial, data, inp, post, prop, iterations, num_chains):
    """
    Performs the MCMC sampling with convergence 
    testing using the imported Gelman_Rubin Statistics

   Parameters:
   ------------
   initial : numpy array, initial conditions of parameters.
   data : numpy array, data.
   post: function, posterior function.
   prop : function, proposal function.
   iterations : int, number of times that mcmc alghoritm runs.
   num_chains : int, number of chains that mcmc alghoritm runs.

   Retruns:
   ---------
   chain_var1_arr : list, chains for variable 1
   chain_var2_arr : list, chains for variable 2
   R1 : float, R value for parameter 1
   R2 : float, R value for parameter 2

    """

    chain_var1 = []
    chain_var2 = []
    prob_l = []
    for _ in range(num_chains):
            chain, prob = mcmc(initial, data, inp, post, prop, iterations)
            var1 = chain[:,0]
            var2 = chain[:,1]
            chain_var1.append(var1)
            chain_var2.append(var2)
            prob_l.append(prob)

    R1 = Gelman_Rubin(chain_var1)    
    R2 = Gelman_Rubin(chain_var2)    
    
    return chain_var1, chain_var2, prob, R1, R2

def proposal(x):
    return np.random.normal(0, 1, 2) + x

def prior_coa(var):
    """
    Prior for coa_phase

    Parameters:
    -----------
    var : float, value for coa_phase

    Returns:
    -------
    int, zero or one
    """
    if 1.5<var<(1.5)*np.pi:
        return 1
    else:
        return 0

def prior_inc(var):
    """
    Prior for inclanation

    Parameters:
    -----------
    var : float, value for inclanation

    Returns:
    -------
    int, zero or one
    """
    if 2.0<var<(1.5)*np.pi:
        return 1
    else:
        return 0
    
def prior_emcee(var):
    """
    Common prior for emcee implementation

    Parameters:
    -----------
    var : dic, values for parameters

    Returns:
    -------
    int, zero or one
    """
    if 2.0<var<(1.
    coa = var['coa_phase']
    inc = var['inclination']
    if 1.5<coa<(1.5)*np.pi and 2.0<inc<(1.5)*np.pi:
        return 1
    else:
        return 0

def log_likelihood_hw(data, inp, var, sigma=1):
    """
    Calculates the log likelihood for handwritten
    version.

    Parameters:
    -----------
    data : numpy array, data.
    inp : dic, parameters for waveform generation
    var : dic, values for parameters

    Returns:
    ---------
    likelihood : float, log likelihood value.
    """
    inp.update({'coa_phase':var[0], 'inclination':var[1]})
    hp, _ = get_fd_waveform(**inp)
    predicted_data = np.real(hp.data)*10**24
    errors = data - predicted_data 
    likelihood = -0.5 * np.sum(errors**2) / sigma**2
    return likelihood 



def posterior(data, inp, var, sigma=1):
    """
    Calculates the posterior.

    Parameters:
    -----------
    data : numpy array, data.
    inp : dic, parameters for waveform generation
    var : dic, values for parameters

    Returns:
    --------
    post : float, posterior
    """
    post = log_likelihood_hw(data, inp, var) + np.log(prior_coa(var[0])) + np.log(prior_inc(var[1]))
    return post

