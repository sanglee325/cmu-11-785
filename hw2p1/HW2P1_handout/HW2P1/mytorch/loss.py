import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        error  = (self.A - self.Y)**2 # TODO
        L      = np.sum(error) / (N *C)
        
        return L
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones   = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")

        self.softmax = np.exp(A)/(np.exp(A) @ Ones) # TODO
        crossentropy = -self.Y * np.log(self.softmax) # TODO
        sum_crossentropy = Ones_N.T @ crossentropy @ Ones
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y # TODO
        
        return dLdA
