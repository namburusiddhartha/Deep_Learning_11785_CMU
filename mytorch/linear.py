import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features, weight_init_fn, bias_init_fn, debug = False):
    
        # self.W    = np.zeros((out_features, in_features), dtype="f")
        # self.b    = np.zeros((out_features, 1), dtype="f")
        self.W = weight_init_fn(out_features, in_features)
        self.b = bias_init_fn(out_features)
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug

    def forward(self, A):
    
        self.A    = A
        self.N    = A.shape[0]
        self.Ones = np.ones((self.N,1), dtype="f")
        Z         = (A @ self.W.T) + (self.Ones @ self.b.T)
        
        return Z
        
    def backward(self, dLdZ):
    
        dZdA      = self.W.T
        dZdW      = self.A
        dZdi      = None #self.b
        dZdb      = self.Ones
        dLdA      = dLdZ @ dZdA.T
        dLdW      = dLdZ.T @ dZdW
        dLdi      = None #dLdZ @ dZdi
        dLdb      = dLdZ.T @ dZdb
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi
        
        return dLdA