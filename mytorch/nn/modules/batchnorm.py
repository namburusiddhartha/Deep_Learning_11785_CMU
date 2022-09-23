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
        
        if eval:
            # TODO
            self.BZ = self.BW * Z + self.Bb
            return self.BZ
            
        self.Z         = Z
        self.N         = self.Z.shape[0]
        
        self.M         = np.expand_dims(np.mean(self.Z, 0), 0) 
        self.V         = np.expand_dims(np.var(self.Z, 0), 0) 
        self.NZ        = (self.Z - self.M )/ np.sqrt(self.V + self.eps)
        self.BZ        = self.BW * self.NZ + self.Bb
        
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.NZ, 0)
        self.dLdBb  = np.sum(dLdBZ, 0)
        
        dLdNZ       = dLdBZ * self.BW
        dLdV        = -0.5*np.expand_dims(np.sum(dLdNZ * (self.Z - self.M) * (self.V + self.eps)**(-3/2), 0), 0)
        dLdM        = -np.expand_dims(np.sum(dLdNZ * (self.V + self.eps)**(-1/2) , 0), 0) - 0.5*np.expand_dims(np.sum(dLdNZ * (self.Z - self.M) * (self.V + self.eps)**(-3/2), 0), 0)*(-2/self.N)*(np.expand_dims(np.sum(self.Z - self.M , 0), 0))
        
        dLdZ        = dLdNZ * (self.V + self.eps)**(-1/2) + dLdV * ((2/self.N)*(self.Z - self.M)) + (1/self.N)*dLdM
        
        return dLdZ