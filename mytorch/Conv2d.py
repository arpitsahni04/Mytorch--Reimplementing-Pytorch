import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        out_width = self.A.shape[2]-self.kernel_size +1
        out_height = self.A.shape[3]-self.kernel_size +1
        Z = np.zeros((self.A.shape[0],self.out_channels,out_width,out_height))  # TODO
        
        for x in range(out_width):
            for y in range(out_height):
                Z[:,:,x,y] =np.tensordot(self.A[:,:,x:x+self.kernel_size,y:y+self.kernel_size],
                                         self.W,axes= [(1,2,3),(1,2,3)])

        Z +=self.b.reshape(1,Z.shape[1],1,1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        for x in range(self.W.shape[2]):
            for y in range(self.W.shape[3]):
                self.dLdW[:,:,x,y] =np.tensordot(dLdZ,self.A[:,:,
                                                        x:x+dLdZ.shape[2],y:y+dLdZ.shape[3]]
                                                            ,axes = [(0,2,3),(0,2,3)]) 
        self.dLdb = np.sum(dLdZ,axis = (0,2,3))  # TODO
        pad_h = self.W.shape[2]-1
        pad_w = self.W.shape[3]-1
        dLdZ_padded = np.pad(dLdZ,((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)))
        dLdA = np.zeros_like(self.A)
        W_flipped = self.W[:,:,::-1,::-1]
        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                dLdA[:,:,i,j]=np.tensordot(dLdZ_padded[:,:,i:i+self.kernel_size,
                                                       j:j+self.kernel_size],W_flipped,axes = [(1,2,3),(0,2,3)])

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,
                 kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        A = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(A)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO

        # Call Conv1d_stride1 backward
        dLdA = dLdZ = self.downsample2d.backward(dLdZ) # TODO
        dLdA = self.conv2d_stride1.backward(dLdZ)
        return dLdA
