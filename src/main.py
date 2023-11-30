import numpy as np
from mcmc import mcmc_with_convergence
import h5py
from matplotlib import pyplot as plt
from pycbc.waveform.waveform import get_fd_waveform
from create_data import create_data
import emcee

# Read the parameters and create the data
param_data = h5py.File('parameter.h5', 'r')
hp, hc, inp = create_data(param_data)
hp_real = np.real(hp.data.data)*10**24
dim = hp_real.shape[0]
noise = np.random.normal(0, 1, dim)
noisy_hp = hp_real + noise

def post(data, var, sigma=1):
    inp.update({'coa_phase':var[0], 'inclination':var[1]})
    hp, hc = get_fd_waveform(**inp)
    predicted_data = np.real(hp.data)*10**24
    errors = data - predicted_data 
    likelihood = -0.5 * np.sum(errors**2) / sigma**2
    return likelihood + np.log(prior_coa(var[0])) + np.log(prior_inc(var[1]))

def proposal(x):
    return np.random.normal(0, 1, 2) + x

def prior_coa(var):
    if 1.5<var<(1.5)*np.pi:
        return 1
    else:
        return 0

def prior_inc(var):
    if 2.0<var<(1.5)*np.pi:
        return 1
    else:
        return 0

initial_guess = np.array([3.5, 4])
num_chains = 5
iterations = 1000

chain_var1, chain_var2, R1, R2 = mcmc_with_convergence(initial_guess, noisy_hp, post, proposal, iterations, num_chains)

print('R1:', R1)
print('R2:', R2)


def prior_emcee(var):
    coa = var['coa_phase']
    inc = var['inclination']
    if 1.5<coa<(1.5)*np.pi and 2.0<inc<(1.5)*np.pi:
        return 1
    else:
        return 0
    
def likelihood_emcee(data, var, sigma=1):
    inp.update(var)
    hp, hc = get_fd_waveform(**inp)
    predicted_data = np.real(hp.data)*10**24
    errors = data - predicted_data 
    likelihood = -0.5 * np.sum(errors**2) / sigma**2
    return likelihood

def post(var, data):
    return likelihood_emcee(data, var) + np.log(prior_emcee(var))

ndim = 2  # Number of parameters
nwalkers = 4  # Number of walkers

sampler = emcee.EnsembleSampler(nwalkers, ndim, post, parameter_names=['coa_phase', 'inclination'], args=[noisy_hp])

initial_guess = np.array([3.5, 4])
nsteps = 1000

initial_positions = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(initial_positions, nsteps, progress=True)

samples = sampler.get_chain(flat=True)


plt.figure()
plt.title("coa_phase Emcee")
plt.plot(samples[:,0])
plt.ylabel('x-value')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("inc Emcee")
plt.plot(samples[:,1])
plt.ylabel('x-value')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("handwritten coa")
plt.plot(chain_var1[0])
plt.ylabel('y-value')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("handwritten inc")
plt.plot(chain_var1[1])
plt.ylabel('y-value')
plt.xlabel('Iteration')
plt.show()