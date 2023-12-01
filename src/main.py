import numpy as np
import mcmc
import h5py
from matplotlib import pyplot as plt
from create_data import create_data
import emcee
from pycbc.waveform.waveform import get_fd_waveform

# Read the parameters and create the data
param_data = h5py.File('parameter.h5', 'r')
hp, hc, inp = create_data(param_data)
hp_real = np.real(hp.data.data)*10**24 # Make signal louder
dim = hp_real.shape[0]
noise = np.random.normal(0, 1, dim)
noisy_hp = hp_real + noise # Adding noise

initial_guess = np.array([3.5, 4])
num_chains = 5
iterations = 10000

chain_var1, chain_var2, prob, R1, R2 = mcmc.mcmc_with_convergence(initial_guess, noisy_hp, inp, mcmc.posterior, mcmc.proposal, iterations, num_chains)

def likelihood_emcee(data, var, sigma=1):
    inp.update(var)
    hp, _ = get_fd_waveform(**inp)
    predicted_data = np.real(hp.data)*10**24
    errors = data - predicted_data 
    likelihood = -0.5 * np.sum(errors**2) / sigma**2
    return likelihood

def post(var, data):
    return likelihood_emcee(data, var) + np.log(mcmc.prior_emcee(var))


ndim = 2  # Number of parameters
nwalkers = 4  # Number of walkers

sampler = emcee.EnsembleSampler(nwalkers, ndim, post, parameter_names=['coa_phase', 'inclination'], args=[noisy_hp])

initial_guess = np.array([3.5, 4])
nsteps = 10000

initial_positions = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(initial_positions, nsteps, progress=True)

samples = sampler.get_chain(flat=True)

print('R1:', R1)
print('R2:', R2)
print('autocorr time', sampler.get_autocorr_time())

plt.figure()
plt.title("coa_phase by using Emcee")
plt.plot(samples[:,0])
plt.ylabel('rad')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("inclination by using Emcee")
plt.plot(samples[:,1])
plt.ylabel('rad')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("coa_phase by using handwritten mcmc")
plt.plot(chain_var1[0])
plt.ylabel('rad')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("inclination by using handwritten mcmc")
plt.plot(chain_var2[0])
plt.ylabel('rad')
plt.xlabel('Iteration')
plt.show()

plt.figure()
plt.title("posterior samples of coa_phase by using Emcee")
plt.hist(samples[:,0])
plt.show()

plt.figure() 
plt.title("posterior samples of inclination by using Emcee")
plt.hist(samples[:,1])
plt.show()

plt.figure()
plt.title("posterior samples of coa_phase by using handwritten mcmc")
plt.hist(chain_var1[0])
plt.show()

plt.figure()
plt.title("posterior samples of inclination by using handwritten mcmc")
plt.hist(chain_var2[0])
plt.show()