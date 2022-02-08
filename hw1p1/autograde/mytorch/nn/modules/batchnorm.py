import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None # TODO
        self.NZ        = None # TODO
        self.BZ        = None # TODO

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
        
        if eval:
            self.NZ        = (Z - self.running_M) / np.sqrt(self.running_V+self.eps)
            self.BZ        = self.BW * self.NZ + self.Bb
            return self.BZ
            
        self.Z         = Z
        self.N         = self.Z.shape[0] # TODO
        
        self.M         = (1/self.N) * np.sum(self.Z, axis=0) # TODO
        self.V         = (1/self.N) * np.sum((self.Z - self.M)**2, axis=0) # TODO
        self.NZ        = (self.Z - self.M) / np.sqrt(self.V+self.eps) # TODO
        self.BZ        = self.BW * self.NZ + self.Bb # TODO
        
        self.running_M = self.alpha * self.running_M + (1-self.alpha)*self.M  # TODO
        self.running_V = self.alpha * self.running_V + (1-self.alpha)*self.V  # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.BZ, axis=0) # TODO
        self.dLdBb  = np.sum(dLdBZ, axis=0 ) # TODO
        
        dLdNZ       = dLdBZ * self.BW # TODO
        dLdV        = (-1/2) * np.sum(dLdNZ * (self.Z - self.M) * (self.V + self.eps)**(-3/2), axis=0) # TODO
        dLdM        =  -1 * np.sum(dLdNZ * (self.V + self.eps)**(-1/2), axis=0) + (-2 / self.N) * dLdV * np.sum(self.Z-self.M, axis=0) # TODO
        
        dLdZ        = dLdNZ * (self.V + self.eps)**(-1/2) + dLdV*2*(self.Z - self.M)/self.N + dLdM / self.N # TODO
        
        return  dLdZ