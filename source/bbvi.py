import numpy as np
import torch
from bvi import BVI

def projection_on_simplex(x):
    u = torch.sort(x, descending=True)[0]
    indices = torch.arange(1, u.shape[0] + 1)
    rho_nz = u + 1. / indices * (1. - torch.cumsum(u, dim=0)) > 0
    rho = indices[rho_nz].max()
    lmbda = 1. / rho * (1. - u[:rho].sum())
    out = torch.max(
        torch.stack([x + lmbda, torch.zeros_like(x, dtype=torch.float32)]), dim=0).values
    return out / out.sum()

class BBVI(BVI):
    
    def __init__(
        self, logp, component_dist, lmb = lambda itr : 1,
        n_samples = 100, n_simplex_iters = 3000, eps = None, **kw
    ):
        super().__init__(component_dist, **kw)
        self.logp = logp
        self.lmb = lmb
        self.n_samples = n_samples
        self.n_simplex_iters = n_simplex_iters
        self.eps = eps
    
    def _compute_weights(self):
        if self.params.shape[0] == 1:
            return torch.tensor([1.])
        else:
            weights = torch.ones(self.params.shape[0], dtype=torch.float32)
            weights /= weights.sum()

            for j in range(self.n_simplex_iters):
                weights.requires_grad_()
                optimizer = torch.optim.SGD([weights], lr=self.lr)

                optimizer.zero_grad()
                self._kl_estimate(self.params, weights).backward()
                optimizer.step()

                weights = projection_on_simplex(weights.detach())

        return weights

    def _error(self):
        return "KL Divergence", self._kl_estimate(self.params, self.weights)
    
    def _objective(self, x, itr):
        h_samples = self.component_dist.generate_samples_for_one_component(x, self.n_samples)
        
        # compute log target density under samples
        lf = self.logp(h_samples).mean()
        
        # compute current log mixture density
        if self.weights.shape[0] > 0:
            lg = self.component_dist.log_pdf(self.params, h_samples)
            if len(lg.shape) == 1:
                # need to add a dimension so that each sample corresponds to a row in lg
                lg = lg[:, None] 
            
            lg = lg[:, self.weights > 0] + torch.log(self.weights[self.weights > 0])
            if self.eps:
                lg = torch.cat((lg, np.log(self.eps) * torch.ones_like((lg.shape[0], 1))), 1)
            lg = torch.logsumexp(lg, dim=1).mean()
        else:
            lg = 0.
        lh = self.component_dist.log_pdf(x, h_samples).mean()

        # print(lg)
        # print(lh)
        # print(lf)

        return lg + self.lmb(self.weights.shape[0]) * lh - lf
    
    def _kl_estimate(self, params, weights):
        out = 0.
        for k in range(weights.shape[0]):
            samples = self.component_dist.generate_samples_for_one_component(params[k, :], self.n_samples)
            lg = self.component_dist.log_pdf(params, samples)
            if len(lg.shape) == 1:
                lg = lg[:,np.newaxis]
            
            lg = torch.logsumexp(lg[:, weights > 0] + torch.log(weights[weights > 0]), dim=1)
            lf = self.logp(samples)
            out += weights[k] * (lg.mean() - lf.mean())
        return out
    
    def _print_perf_w(self, itr, x, obj, grd):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        print("{:^30}|{:^30}|{:^30.3f}|{:^30.3f}".format(itr, str(x), np.sqrt((grd**2).sum()), obj))

    def _get_mixture(self):
        #just get the unflattened params and weights; for KL BVI these correspond to mixture components
        output = self.component_dist.unflatten(self.params)
        output.update([('weights', self.weights)])
        return output
