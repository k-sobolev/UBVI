import torch
import numpy as np

from matplotlib import pyplot as plt
from source.gaussians import Gaussian
from source.ubvi import UBVI

experiment = lambda x: -((x - 1) ** 2).sum(dim=-1) / 2
test = UBVI(experiment, Gaussian(1), num_opt_steps=1000, n_samples=1000, n_init=100, init_inflation=100)

test.build(20)

samples = test._sample_g(100000).detach().numpy().flatten()

plt.figure(figsize=(10, 10))

plt.hist(samples, density=True, bins=120, range=[-10, 10])

grid = np.linspace(-10, 10, 100)
plt.plot(grid, (1 / np.pi) / (1 + grid ** 2))

plt.title('Boosting Cauchy with gaussians')
plt.ylabel(r'$p$')
plt.xlabel(r'$x$')
plt.savefig('/results/ubvi/test.jpg')
plt.show()


