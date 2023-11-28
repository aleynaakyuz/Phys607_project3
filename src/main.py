import numpy as np
import mcmc
import h5py
from matplotlib import pyplot as plt
from pycbc.waveform.waveform import get_fd_waveform
from create_data import create_data

# Read the parameters and create the data
param_data = h5py.File('Phys607_project3/src/data/parameter.h5')
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

itt = 50000
inital_guess = np.array([3.5, 4])

chain, prob = mcmc(inital_guess, noisy_hp, post, proposal, itt)

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