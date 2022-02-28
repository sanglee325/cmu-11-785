import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.Z         = Z
        if eval:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.NZ * self.BW + self.Bb
            return self.BZ
            
        
        self.N         = Z.shape[0] # TODO
        
        self.M         = np.mean(self.Z, axis=0) # TODO
        self.V         = np.var(self.Z, axis=0) # TODO
        self.NZ        = (self.Z - self.M) / np.sqrt(self.V + self.eps) # TODO
        self.BZ        = self.NZ * self.BW + self.Bb # TODO
        
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        sqrt_var_plus_eps = np.sqrt(self.V + self.eps)
        b = dLdBZ.shape[0]
        
        dLdNZ       = self.BW * dLdBZ # TODO
        self.dLdBb  = np.sum(dLdBZ, axis=0, keepdims=True) # TODO
        self.dLdBW  = np.sum(dLdBZ * self.BZ, axis=0, keepdims=True) # TODO

        dLdV = -0.5 * np.sum((dLdNZ * (self.Z - self.M) / (sqrt_var_plus_eps ** 3)), axis=0) # TODO
        first_term_dmu = -1 * np.sum((dLdNZ / sqrt_var_plus_eps), axis=0)
        second_term_dmu = -((2*self.dLdBW )/b) * np.sum((self.Z - self.M), axis=0)
        dLdM = first_term_dmu + second_term_dmu # TODO
        
        first_term_dZ = dLdNZ/sqrt_var_plus_eps
        second_term_dZ = dLdV * (2/b) * (self.Z - self.M)
        third_term_dZ = dLdM * (1/b)

        dLdZ        = first_term_dZ + second_term_dZ + third_term_dZ # TODO
        
        return  dLdZ