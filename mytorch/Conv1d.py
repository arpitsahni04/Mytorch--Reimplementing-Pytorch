# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        out_size = self.A.shape[2] - self.kernel_size + 1
        Z = np.zeros((self.A.shape[0],self.out_channels, out_size))  # TODO
        for i in range(out_size):
            Z[:,:,i] = np.tensordot(self.A[:,:,i:i+self.kernel_size],self.W,axes=[(1,2),(1,2)])
        # import pdb 
        # pdb.set_trace()
        Z+=self.b.reshape(1,Z.shape[1],1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        for i in range(self.W.shape[2]):
            self.dLdW[:,:,i] = np.tensordot(dLdZ,self.A[:,:,i:i+dLdZ.shape[2]],axes=((0,2),(0,2))) # TODO
        self.dLdb = np.sum(dLdZ,axis=(2,0)) # TODO # doubt
        dLdA = np.zeros_like(self.A)  # TODO
        pad =self.W.shape[2]-1

        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad)), 'constant', constant_values=0)
        W_flipped = self.W[:,:,::-1]
        # print(self.W.shape)
        # print(W_flipped.shape)
        for i in range(self.A.shape[2]):
            dLdA[:,:,i] = np.tensordot(dLdZ_padded[:,:,i:i+self.W.shape[2]],W_flipped,axes = [(1,2),(0,2)])
        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                 weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z =self.downsample1d.forward(Z)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
         # TODO
        dLdA = self.conv1d_stride1.backward(dLdZ)
        return dLdA
