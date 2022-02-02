import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features, debug = False):
    
        self.W    = np.zeros((out_features, in_features), dtype="f")
        self.b    = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug

    def forward(self, A):
    
        self.A    = A
        self.N    = A.shape[0]
        self.Ones = np.ones((self.N,1), dtype="f")
        Z         = np.dot(self.A, self.W.T) + self.b.T
        
        return Z
        
    def backward(self, dLdZ):
    
        dZdA      = self.W.T # TODO
        dZdW      = self.A # TODO
        dZdi      = None
        dZdb      = self.Ones # TODO
        dLdA      = np.dot(dLdZ, dZdA.T) # TODO
        dLdW      = np.dot(dLdZ.T, dZdW) # TODO
        dLdi      = None
        dLdb      = np.dot(dLdZ.T, dZdb) # TODO
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi
        
        return NotImplemented