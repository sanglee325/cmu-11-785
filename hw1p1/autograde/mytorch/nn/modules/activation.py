import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = 1/(1 + np.exp(-Z))
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A * (1 - self.A)
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.sinh(Z)/np.cosh(Z)
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - self.A * self.A
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = np.maximum(0, Z)
        
        return self.A
    
    def backward(self):
    
        dAdZ = (self.A > 0).astype(int)
        
        return dAdZ
        
        
