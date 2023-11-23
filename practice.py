
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data from HDF file
data = h5py.File('./src/data/parameter.h5', 'r')
distance_data = data['distance'][:]
data.close()
print('distance data:', distance_data)
# Define the target distribution function
def f(distance):
    # Implement distribution function based on the data
    mu = np.mean(distance_data) #mean
    sigma = np.std(distance_data) #standard deviation
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((distance - mu) / sigma)**2) #probability distribution

# Metropolis-Hastings algorithm
def metropolis_hastings(initial_guess, num_iterations):
    samples = [initial_guess]
    num_accept = 0

    for i in range(num_iterations):
        # Sample candidate from normal distribution
        candidate = np.random.normal(samples[-1], 4)

        # Calculate probability of accepting this candidate
        prob = min(1, f(candidate) / f(samples[-1]))

        # Accept with the calculated probability
        if np.random.random() < prob:
            samples.append(candidate)
            num_accept += 1

    return np.array(samples)

# Set initial guess and number of iterations
initial_guess = 40000  # this should be based on prior knowledge or so
num_iterations = 1000000

# Run Metropolis-Hastings algorithm
distance_samples = metropolis_hastings(initial_guess, num_iterations)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(distance_samples, label='MCMC samples')
# plt.axhline(np.mean(distance_data), color='red', linestyle='--', label='True mean')
plt.xlabel('Iterations')
plt.ylabel('Parameter (Distance)')
plt.legend()
plt.show()

