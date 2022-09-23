import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        
        #print(h.shape)
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r0 = self.Wrx @ self.x + self.brx + self.Wrh @ self.hidden + self.brh
        self.r = self.r_act(self.r0)
        
        self.z0 = self.Wzx @ self.x + self.bzx + self.Wzh @ self.hidden + self.bzh
        self.z = self.z_act(self.z0)
        
        self.n0 = self.Wnh @ self.hidden + self.bnh
        self.n1 = self.Wnx @ self.x + self.bnx + self.r * (self.n0)
        self.n = self.h_act(self.n1)
        
        h_t = ((self.z) * self.hidden) + ((1 - self.z) * self.n)
        
        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # This code should not take more than 25 lines.
        x = np.expand_dims(self.x, 1)
        h = np.expand_dims(self.hidden, 1)
                
        self.dz = delta * np.expand_dims((self.hidden - self.n), 0)
        self.dn = delta * np.expand_dims((1 - self.z), 0)
        
        n0 = self.Wnh @ self.hidden + self.bnh
        n1 = self.Wnx @ self.x + self.bnx + self.r * (n0)
        self.dWnx = (self.dn * np.expand_dims(self.h_act.derivative(), 0)).T @ x.T
        self.dbnx = (self.dn * np.expand_dims(self.h_act.derivative(), 0))
        
        self.dr = self.dn * self.h_act.derivative() * (self.Wnh @ self.hidden + self.bnh)
        
        self.dh_t = delta * self.z + (self.dn * np.expand_dims(self.h_act.derivative(), 0) * self.r) @ self.Wnh + (self.dz * np.expand_dims(self.z_act.derivative(), 0)) @ self.Wzh + (self.dr * np.expand_dims(self.r_act.derivative(), 0)) @ self.Wrh        
        
        self.dx = (self.dn * np.expand_dims(self.h_act.derivative(), 0)) @ self.Wnx + (self.dz * np.expand_dims(self.z_act.derivative(), 0)) @ self.Wzx + (self.dr * np.expand_dims(self.r_act.derivative(), 0)) @ self.Wrx

        
        self.dWnh = (self.dn * np.expand_dims(self.h_act.derivative(), 0) * self.r).T @ h.T     
        self.dbnh = (self.dn * np.expand_dims(self.h_act.derivative(), 0) * self.r)
        
        self.dWzx = (self.dz * np.expand_dims(self.z_act.derivative(), 0)).T @ x.T
        self.dbzx = np.squeeze(self.dz * np.expand_dims(self.z_act.derivative(), 0))
        
        self.dWzh = (self.dz * np.expand_dims(self.z_act.derivative(), 0)).T @ h.T
        self.dbzh = np.squeeze(self.dz * np.expand_dims(self.z_act.derivative(), 0))
        
        self.dWrh = (self.dr * np.expand_dims(self.r_act.derivative(), 0)).T @ h.T
        self.dbrh = np.squeeze(self.dr * np.expand_dims(self.r_act.derivative(), 0))
        
        self.dWrx = (self.dr * np.expand_dims(self.r_act.derivative(), 0)).T @ x.T
        self.dbrx = np.squeeze(self.dr * np.expand_dims(self.r_act.derivative(), 0))
        
        dx = self.dx
        dh = self.dh_t
        
        

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        
