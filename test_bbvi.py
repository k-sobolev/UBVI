import torch
import numpy as np

from matplotlib import pyplot as plt
from source.gaussians import Gaussian
from source.bbvi import BBVI

experiment = lambda x: -((x - 1) ** 2).sum(dim=-1) / 2
test = BBVI(experiment, Gaussian(1), num_opt_steps=5000, n_samples=1000, n_init=500, lmb=lambda x: 10)

test.build(20)

def sampler(test,n_samples=100000):
    mus = test._get_mixture()['mus']
    sigmas = test._get_mixture()['sigmas']
    weights = test._get_mixture()['weights']
    samples=[]
    for _ in range(n_samples):
        i = np.random.multinomial(1, weights).argmax()
        samples.append(np.random.normal(mus[i,0].detach().numpy(),sigmas[i,0].detach().numpy(),1))#[0]
    return np.array(samples)[:,0]

samples = sampler(test)

plt.figure(figsize=(10, 10))

plt.hist(samples, density=True, bins=120, range=[-10, 10])

grid = np.linspace(-10, 10, 100)
plt.plot(grid, (1 / np.pi) / (1 + grid ** 2))

plt.title('Boosting Cauchy with gaussians')
plt.ylabel(r'$p$')
plt.xlabel(r'$x$')
plt.savefig('/results/bbvi/test.jpg')
plt.show()


