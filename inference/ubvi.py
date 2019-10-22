import numpy as np
from scipy.optimize import nnls 
from bvi import BVI


def logsumexp(input):
	return torch.log(torch.sum(torch.exp(input)))

class UBVI(BVI):

	def __init__(self, logp, component_dist, opt_alg, n_samples = 100, n_logfg_samples = 100, **kw):
        super().__init__(component_dist, opt_alg, **kw)
        self.logp = logp
        self.n_samples = n_samples
        self.n_logfg_samples = n_logfg_samples
        self.Z = torch.empty((0,0))
        self._logfg = torch.empty(0)
        self._logfgsum = -np.inf
        
    def _compute_weights(self):
        #compute new row/col for Z
        Znew = np.exp(self.component_dist.log_sqrt_pair_integral(self.params[-1, :], self.params))

        #expand Z
        Zold = self.Z
        self.Z = torch.zeros((self.params.shape[0], self.params.shape[0]))
        self.Z[:-1, :-1] = Zold
        self.Z[-1, :] = Znew
        self.Z[:, -1] = Znew

        #expand logfg
        logfgold = self._logfg
        self._logfg = torch.zeros(self.params.shape[0])
        self._logfg[:-1] = logfgold
        self._logfg[-1] = self._logfg_est(self.params[-1, :])
        
        #compute optimal weights via nnls
        if self.params.shape[0] == 1:
            w = torch.Tensor([1])
        else:
            Linv = torch.inverse(torch.cholesky(self.Z))
            d = torch.exp(self._logfg-self._logfg.max()) #the opt is invariant to d scale, so normalize to have max 1
            b = torch.Tensor(nnls(Linv.numpy(), - torch.dot(Linv, d).numpy())[0])
            lbd = torch.dot(Linv, b + d)
            w = torch.max(0., torch.dot(Linv.T, lbd/torch.sqrt(((lbd**2).sum()))))

        #compute weighted logfg sum
        #self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg + np.log(np.maximum(w, 1e-64)))))
        self._logfgsum = logsumexp(torch.cat((-np.inf, self._logfg[w>0] + torch.log(w[w>0])), 1))
       
        #return the weights
        return w
        
    def _error(self):
        return "Hellinger Dist Sq", self._hellsq_estimate()

    def _hellsq_estimate(self):
        samples = self._sample_g(self.n_samples)
        lf = 0.5 * self.logp(samples)
        lg = self._logg(samples)
        ln = np.log(self.n_samples)
        return 1. - torch.exp(logsumexp(lf-lg-ln) - 0.5 * logsumexp(2*lf-2*lg-ln))
    
    def _objective(self, x, itr):
    	
        allow_negative = False if itr < 0 else True
        
        #lgh = -np.inf if len(self.weights) == 0 else logsumexp(np.log(np.maximum(self.weights[-1], 1e-64)) + self.component_dist.log_sqrt_pair_integral(x, self.params))
        # lgh = -np.inf if self.weights.size == 0 else logsumexp(torch.log(self.weights[self.weights>0]) + np.atleast_2d(self.component_dist.log_sqrt_pair_integral(x, self.params))[:, self.weights>0])
        lgh = -np.inf if self.weights.size == 0 else logsumexp(torch.log(self.weights[self.weights>0]) + (self.component_dist.log_sqrt_pair_integral(x, self.params))[:, self.weights>0])
        h_samples = self.component_dist.sample(x, self.n_samples)
        lf = 0.5*self.logp(h_samples)
        lh = 0.5*self.component_dist.logpdf(x, h_samples) 
        ln = torch.log(self.n_samples)
        lf_num = logsumexp(lf - lh - ln)
        lg_num = self._logfgsum + lgh 
        log_denom = 0.5*torch.log(1.-torch.exp(2*lgh))
        if lf_num > lg_num:
            logobj = lf_num - log_denom + torch.log(1.-torch.exp(lg_num-lf_num))
            neglogobj = -logobj
            return neglogobj
        else:
            if not allow_negative:
                return np.inf
            lognegobj = lg_num - log_denom + torch.log(1.-torch.exp(lf_num-lg_num))
            return lognegobj

    def _logfg_est(self, param):
        samples = self.component_dist.sample(param, self.n_samples)
        lf = 0.5*self.logp(samples)
        lg = 0.5*self.component_dist.logpdf(param, samples)
        ln = torch.log(self.n_samples)
        return logsumexp(lf - lg - ln)
    
    def _logg(self, samples):
        logg_x = 0.5 * self.component_dist.logpdf(self.params, samples)
        if len(logg_x.shape) == 1:
            logg_x = logg_x[:,np.newaxis]
        #return logsumexp(logg_x + np.log(np.maximum(self.weights[-1], 1e-64)), axis=1)
        return torch.logsumexp(logg_x[:, self.weights>0] + torch.log(self.weights[self.weights > 0]), dim=1)
        
    def _sample_g(self, n):
        #samples from g^2
        g_samples = torch.zeros((n, self.component_dist.d))
        #compute # samples in each mixture pair
        g_ps = (self.weights[:, np.newaxis]*self.Z*self.weights).flatten()
        g_ps /= g_ps.sum()
        pair_samples = torch.multinomial(g_ps, n).view(self.Z.shape)
        #symmetrize (will just use lower triangular below for efficiency)
        pair_samples = pair_samples + pair_samples.T
        for k in range(self.Z.shape[0]):
            pair_samples[k,k] /= 2
        #invert sigs
        #fill in samples
        cur_idx = 0
        for j in range(self.Z.shape[0]):
            for m in range(j+1):
                n_samps = pair_samples[j,m]
                g_samples[cur_idx:cur_idx+n_samps, :] = self.component_dist.cross_sample(self.params[j, :], self.params[m, :], n_samps)
                cur_idx += n_samps
        return g_samples

    def _get_mixture(self):
        #get the mixture weights
        ps = (self.weights[:, np.newaxis]*self.Z*self.weights).flatten() 
        ps /= ps.sum()
        paired_params = torch.zeros((self.params.shape[0]**2, self.params.shape[1]))
        for i in range(self.N): 
            for j in range(self.N): 
                paired_params[i*self.N + j, :] = self.component_dist._get_paired_param(self.params[i, :], self.params[j, :], flatten=True)
        #get the unflattened params and weights
        output = self.component_dist.unflatten(paired_params)
        output.update([('weights', ps)])
        return output