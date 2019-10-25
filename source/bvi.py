import torch
import time

class BVI(object):

    def __init__(
        self, component_dist, n_init = 10, init_inflation = 100, lr=0.01, num_opt_steps=100):

        self.N = 0 # current num of components
        self.component_dist = component_dist # component distribution object
        self.weights = torch.empty(0) # weights
        self.params = torch.empty((0, 0)) # components' parameters
        self.error = float('inf') # error for the current mixture
        self.n_init = n_init # number of initializations for each component
        self.init_inflation = init_inflation # noise multiplier for initializations
        self.lr = lr
        self.num_opt_steps = num_opt_steps

    def build(self, N):
        # build the approximation up to N components
        for i in range(self.N, N):
            error_flag = False

            # initialize the next component
            print("Initializing component " + str(i + 1) + "... ")
            x0 = self._initialize()

            # if this is the first component, set the dimension of self.params
            if self.params.size == 0:
                self.params = torch.empty((0, x0.shape[0]))
            print("Initialization of component " + str(i + 1) + " complete, x0 = " + str(x0))
            
            # build the next component
            print("Optimizing component " + str(i + 1) + "... ")
            # try:
            current_param = x0.clone().detach().requires_grad_()
            optimizer = torch.optim.Adam([current_param], lr=self.lr)
            for j in range(self.num_opt_steps):
                optimizer.zero_grad()
                self._objective(current_param, j).backward(retain_graph=True)
                optimizer.step()

                # print(current_param)

            new_param = current_param
                
            #     if not torch.BoolTensor.all(torch.isfinite(new_param)):
            #         raise

            # except: # bbvi can run into bad degeneracies; if so, just revert to initialization and set weight to 0
            #     error_flag = True
            #     new_param = x0
            print("Optimization of component " + str(i + 1) + " complete")

            # add it to the matrix of flattened parameters
            if self.params.shape[0] == 0:
                self.params = new_param[None]
            else:
                self.params = torch.cat((self.params, new_param[None]), 0)

            # compute the new weights and add to the list
            print('Updating weights...')
            self.weights_prev = self.weights.clone()
            # try:
            self.weights = self._compute_weights()
            #     if not torch.BoolTensor.all(torch.isfinite(self.weights)) or error_flag:
            #         raise
            # except: # bbvi can run into bad degeneracies; if so, just throw out the new component
            #     self.weights = torch.cat((self.weights_prev, 0.), 1)

            print('Weight update complete...')

            # estimate current error
            error_str, self.error = self._error()

            # print out the current error
            print('Component ' + str(self.params.shape[0]) + ':')
            print(error_str +': ' + str(self.error))
            print('Params:' + str(self.component_dist.unflatten(self.params)))
            print('Weights: ' + str(self.weights))
            
        # update self.N to the new # components
        self.N = N

        #generate the nicely-formatted output params
        output = self._get_mixture()
        output['obj'] = self.error
        return output
        
        
    def _initialize(self):
        best_param = None
        best_objective = float('inf')

        # try initializing n_init times
        for n in range(self.n_init):
            current_param = self.component_dist.params_init(
                self.params, self.weights, self.init_inflation)
            current_objective = self._objective(current_param, -1)
            
            if current_objective < best_objective or best_param is None:
                best_param = current_param
                best_objective = current_objective
            if (n == 0 or n == self.n_init - 1):
                if n == 0:
                    print("{:^30}|{:^30}|{:^30}".format('Iteration', 'Best param', 'Best objective'))
                print("{:^30}|{:^30}|{:^30}".format(
                    n, str(best_param), str(best_objective)))
        if best_param is None:
            # if every single initialization had an infinite objective, just raise an error
            raise ValueError
        
        # return the initialized result
        return best_param
    
    def get_weights(self):
        return self.weights

    def _compute_weights(self):
        raise NotImplementedError
        
    def _objective(self, itr):
        raise NotImplementedError
        
    def _error(self):
        raise NotImplementedError
  
    def _get_mixture(self):
        raise NotImplementedError