import numpy as np
import mcmc
from matplotlib import pyplot as plt

def post(x):
    return 1 / (2 * np.pi) ** 0.5 * np.exp(-1/2 * (x[0] ** 2.0 + x[1] ** 2.0))
   
def proposal(x):
    return np.random.normal(0, 1, 2) + x

itt = 1000000

chain, prob = mcmc.mcmc(np.array([10, 10]), post, proposal, itt)

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