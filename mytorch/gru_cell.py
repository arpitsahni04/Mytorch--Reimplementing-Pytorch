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

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        self.r_linear = self.Wrx@self.x + self.brx + self.Wrh@self.hidden + self.brh
        self.r =  self.r_act.forward(self.r_linear)
        self.z_linear = self.Wzx@self.x + self.bzx + self.Wzh@self.hidden + self.bzh
        self.z = self.z_act.forward(self.z_linear)
        self.n_linear = self.Wnx@self.x + self.bnx + self.r *(self.Wnh@self.hidden + self.bnh)
        self.n = self.h_act.forward(self.n_linear)
        h_t = (1-self.z )*self.n + self.z*self.hidden# TODO
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
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

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        self.x = self.x.reshape(-1,1)
        self.hidden = self.hidden.reshape(-1,1)
        self.z = self.z.reshape(-1,1)
        self.r = self.r.reshape(-1,1)
        self.n = self.n.reshape(-1,1)
        delta = delta.reshape(-1,1)
        self.bzh = self.bzh.reshape(-1,1)
        self.bnh = self.bnh.reshape(-1,1)
        self.brh = self.brh.reshape(-1,1)
        self.bnx = self.bnx.reshape(-1,1)
        self.bzx = self.bzx.reshape(-1,1)
        self.brx = self.brx.reshape(-1,1)
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        dz = self.hidden- self.n
        dz =  delta*dz
        dn = delta*(1-self.z)
        der = dn*self.h_act.backward(self.n) * (self.Wnh@self.hidden + self.bnh)
        dr_act = self.r_act.backward(self.r).reshape(-1,1)
        k =(der*dr_act)
        self.dWrh = np.dot(k,self.hidden.T)
        self.dWrx = np.dot(k,self.x.T)
        self.dbrx = k.reshape(-1,) 
        self.dbrh= k.reshape(-1,)
        dz_act = self.z_act.backward(self.z).reshape(-1,1)
        k2 = dz*dz_act
        self.dWzh = np.dot(k2,self.hidden.T)
        self.dWzx=  np.dot(k2,self.x.T)
        self.dbzx=  k2.reshape(-1,)  
        self.dbzh=  k2.reshape(-1,) 
        dh_act = self.h_act.backward(self.n).reshape(-1,1)
        k3 = dn*dh_act
        self.dWnh = np.dot(self.r*k3, self.hidden.T)
        self.dWnx =  np.dot(k3,self.x.T)
        self.dbnx =  k3.reshape(-1,)
        self.dbnh =  (k3*self.r).reshape(-1,)
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order,pand the
        #    local autograder will tell you which gradient you are currently failing.
        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        dx = k3.T@self.Wnx + k.T@self.Wrx + k2.T@self.Wzx
        dh_prev_t = (k3*self.r).T@self.Wnh + (delta*self.z).T + k2.T @ self.Wzh + k.T@self.Wrh
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
        # raise NotImplementedError
