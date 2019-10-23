import torch
import math

from numpy.random import choice

class Gaussian(object):
    def __init__(self, d):
        # d is dimension of the random variable space
        # we only consider diagonal covariation matrices
        self.d = d

    def _atleast_2d(self, tensor):
        if len(tensor.shape) < 2:
            return tensor[None, :]
        return tensor

    def unflatten(self, params):
        params = self._atleast_2d(params)
        N = params.shape[0]
        # first d params are mu values
        mus = params[:, :self.d]
        # following params define the covariance matrix
        log_sigmas = params[:, self.d:]
        return {"mus": mus, "sigmas": torch.exp(log_sigmas)}
        
    def log_pdf(self, params, X):
        theta = self.unflatten(params)
        if len(X.shape) == 1 and self.d == 1:
            # in one-dimensional case need to add a dimension so that each row is an observation
            X = X[:, None]
        X = self._atleast_2d(X)
        
        mus = theta['mus']
        sigmas = theta['sigmas']
        
        log_pdf = -0.5 * mus.shape[1] * torch.log(torch.tensor(2 * math.pi)) - \
                  0.5 * torch.sum(torch.log(sigmas), axis=1) - \
                  0.5 * torch.sum((X[:, None, :] - mus) ** 2 / sigmas, axis=2)
        
        if log_pdf.shape[1] == 1:
            log_pdf = log_pdf[:, 0]
        return log_pdf
            
    def generate_samples_for_one_component(self, param, num_samples):
        epsilons = torch.randn(num_samples, self.d)
        mu = param[:self.d]
        
        log_sigma = param[self.d:]
        std = torch.exp(log_sigma / 2)
        return mu + epsilons * std
        
    def _get_paired_param(self, param_a, param_b, flatten=False):
        theta = self.unflatten(torch.stack((param_a, param_b)))
        mus = theta['mus']
        sigmas = theta['sigmas']

        paired_sigma = 2. / (1 / sigmas[0, :] + 1 / sigmas[1, :])
        paired_mu = 0.5 * paired_sigma * (mus[0, :] / sigmas[0, :] + mus[1, :] / sigmas[1, :])

        if not flatten:
            return paired_mu, paired_sigma
        else: 
            return torch.cat((paired_mu, torch.log(paired_sigma)))
    
    def generate_samples_for_paired_distribution(self, param_a, param_b, num_samples):
        paired_mu, paired_sigma = self._get_paired_param(param_a, param_b)
        
        return paired_mu + torch.sqrt(paired_sigma) * torch.randn((num_samples, self.d))
        
    def log_sqrt_pair_integral(self, new_param, old_params):
        old_params = self._atleast_2d(old_params)
        mu_new = new_param[:self.d]
        mus_old = old_params[:, :self.d]
        
        log_sigma_new = new_param[self.d:]
        log_sigmas_old = old_params[:, self.d:]
        log_sigmas_all = torch.log(torch.tensor(0.5)) + \
                         torch.logsumexp(
                            torch.stack([log_sigma_new.expand_as(log_sigmas_old), log_sigmas_old]),
                            dim=0)

        return -0.125 * torch.sum(torch.exp(-log_sigmas_all) * (mu_new - mus_old) ** 2, axis=1) - \
               0.5 * torch.sum(log_sigmas_all, axis=1) + 0.25 * torch.sum(log_sigma_new) + \
               0.25 * torch.sum(log_sigmas_old, axis=1)

    def params_init(self, params, weights, inflation):
        # initialization with heuristics

        params = self._atleast_2d(params)

        i = params.shape[0]
        if i == 0:
            mu = torch.normal(torch.zeros(self.d), inflation * torch.ones(self.d))
            
            log_sigma = torch.zeros(self.d)
            new_param = torch.cat((mu, log_sigma), dim=0)
        else:
            mus = params[:, :self.d]
            probs = ((weights ** 2) / (weights ** 2).sum()).detach().numpy()
            k = choice(range(i), p=probs)
            
            log_sigmas = params[:, self.d:]
            mu = mus[k,:] + torch.randn(self.d) * torch.sqrt(torch.tensor(inflation, dtype=torch.float32)) * torch.exp(log_sigmas[k, :])
            log_sigma = torch.randn(self.d) + log_sigmas[k, :] 
            new_param = torch.cat((mu, log_sigma), dim=0)
        return new_param
    
    def print_perf(self, itr, x, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format(
                'Iteration', 'mu', 'log_sigma', 'Boosting Obj'
            ))
        if self.diag:
            print("{:^30}|{:^30}|{:^30}|{:^30.3f}".format(
                itr, 
                str(x[:min(self.d, 4)]),
                str(x[self.d: self.d + min(self.d, 4)]),
                obj
                ))
