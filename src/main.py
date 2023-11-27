import numpy as np
import mcmc
import h5py
from matplotlib import pyplot as plt
from pycbc.waveform.waveform import get_fd_waveform

# Read and add noise to the data
data = h5py.File('./data/data.h5')
hp = data['hp'][:]
dim = hp.shape[0]
noise = np.random.normal(0, 1, dim)*10**(-24)
noisy_hp = hp + noise

# Read the parameters
parameters = {
    "mass1": "mass1",
    "mass2": "mass2",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "coa_phase": "coa_phase",
    "inclination": "inclination",
    "distance": "distance",
    "ra": "ra",
    "dec": "dec",
    "polarization": "polarization"
}

parameters2 = {
    "approximant": "IMRPhenomXPHM",
    "delta_f": 0.1,
    "f_lower": 10
}

param_data = h5py.File('Phys607_project3/src/data/parameter.h5')
data_dic = {key: param_data[dataset_key][:][0] for key, dataset_key in parameters.items()}
inp = {**data_dic, **parameters2}

def log_likelihood(data, dist, sigma=0.1):
    inp.update({'distance':dist})
    predicted_data = get_fd_waveform(**inp) + np.random.normal(0, 1, dim)*10**(-24) 
    errors = data - predicted_data
    likelihood = np.log(1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * np.sum(errors**2) / sigma**2))
    return likelihood

def prior():
    pr = np.random.uniform(40000, 50000)  
    return pr

itt = 1000000
inital_guess = 49000

chain, prob = mcmc(inital_guess, noisy_hp, log_likelihood, prior, itt)

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain[:,0])
plt.ylabel('x-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain[:,0])
plt.xlim(0, 100)
plt.ylabel('x-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain[:,1])
plt.ylabel('y-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain[:,1])
plt.xlim(0, 100)
plt.ylabel('y-value')
plt.xlabel('Iteration')