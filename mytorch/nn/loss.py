import numpy as np


class MSELoss:

    def forward(self, A, Y):


        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]  # TODO
        self.C = self.A.shape[1]  # TODO
        se = (self.A-self.Y)*(self.A-self.Y)  # TODO
        sse = np.dot(np.dot(np.ones((1,self.N)),se),np.ones((self.C,1)))  # TODO
        mse = sse/(2*self.N*self.C)  # TODO

        return mse[0][0]

    def backward(self):

        dLdA = (self.A-self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):

        # import pdb
        # pdb.set_trace()
        self.A = A
        self.Y = Y
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1] # TODO

        Ones_C = np.ones((C,1))  # TODO
        Ones_N = np.ones((N,1))  # TODO

        self.softmax = np.exp(A)/np.sum(np.exp(A),axis =1).reshape(-1,1)  # TODO
        crossentropy = -(self.Y * np.log(self.softmax)) @ Ones_C  # TODO
        sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        L = sum_crossentropy / N

        return L[0][0]

    def backward(self):

        dLdA = self.softmax - self.Y  # TODO

        return dLdA
