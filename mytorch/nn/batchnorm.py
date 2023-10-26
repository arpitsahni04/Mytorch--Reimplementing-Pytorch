import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):

        self.Z = Z
        self.N = self.Z.shape[0]  # TODO
        self.M = np.mean(self.Z,axis = 0).reshape(1,-1) # TODO
        self.V = np.var(self.Z,axis =0).reshape(1,-1)  # TODO

        if eval == False:
            # training mode
            self.NZ = (self.Z - (np.ones((self.N,1))@self.M))/ np.sqrt(np.ones(
                (self.N,1))@(self.V + self.eps))# TODO
            self.BZ = (np.ones((self.N,1))@self.BW ) * self.NZ + np.ones((self.N,1))@self.Bb # TODO

            self.running_M = self.alpha *self.running_M + (1-self.alpha)*self.M  # TODO
            self.running_V =self.alpha *self.running_V + (1-self.alpha)*self.V    # TODO
        else:
            # inference mode
            self.NZ = (self.Z - (np.ones((self.N,1))
                                 @self.running_M))/ np.sqrt(np.ones(
                                     (self.N,1))@(self.running_V + self.eps))# TODO
            self.BZ = (np.ones((self.N,1))@self.BW ) * self.NZ + np.ones((self.N,1))@self.Bb # TODO

        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW =np.sum(dLdBZ*self.NZ,axis = 0,keepdims=True)  # TODO
        self.dLdBb = np.sum(dLdBZ,axis=0,keepdims=True)  # TODO
        dLdNZ = dLdBZ *(np.ones((self.N,1))@self.BW)  # TODO
        dLdV = -0.5 *np.sum((dLdNZ * (self.Z-(np.ones((self.N,1))@self.M)) * 
                             np.power((np.ones((self.N,1))@(self.V + self.eps)),-1.5)),
                            axis = 0,keepdims=True)  # TODO
        
        dZdM = -np.power((self.V + self.eps),-0.5) - 0.5*(self.Z - self.M) * np.power((self.V + self.eps),-1.5) * ((-2/self.N)*
                                                                                                                np.sum((self.Z - self.M),axis = 0))
        dLdM = np.sum(dZdM * dLdNZ,axis =0) # TODO
        dLdZ = dLdNZ * np.power((self.V + self.eps),-0.5) + dLdV * ((2/self.N)*(self.Z - (np.ones((self.N,1))@self.M))) + ((1/self.N)*dLdM) # TODO
        return dLdZ
