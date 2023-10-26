import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        import pdb
        self.A = A 
        out_width = A.shape[2]-self.kernel + 1
        out_height = A.shape[3]-self.kernel + 1
        self.pidx=np.zeros((A.shape[0],A.shape[1],out_width,out_height,2))
        Z =  np.zeros((A.shape[0],A.shape[1],out_width,out_height))
        for batch in range(Z.shape[0]):
            for map_idx in range(Z.shape[1]):
                for x in range(out_height):
                    for y in range(out_width):
                        # pdb.set_trace()
                        self.pidx[batch,map_idx,x,y,:] = np.unravel_index(np.argmax(A[batch,map_idx,x:x+self.kernel,y:y+self.kernel]),
                                                                        order ="C",shape= A[batch,map_idx,x:x+self.kernel,y:y+self.kernel].shape)
                        x_max,y_max =self.pidx[batch,map_idx,x,y,:] 
                        Z[batch,map_idx,x,y] =A[batch,map_idx,x+ int(x_max),y+ int(y_max)]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        import pdb
        
        dLdA = np.zeros_like(self.A)
        for batch in range(dLdZ.shape[0]):
            for map_idx in range(dLdZ.shape[1]):
                for x in range(dLdZ.shape[2]):
                    for y in range(dLdZ.shape[3]):
                        x_max,y_max = self.pidx[batch,map_idx,x,y]
                        # pdb.set_trace()
                        
                        dLdA[batch,map_idx,x+ int(x_max),y+ int(y_max)] += np.sum(dLdZ[batch,map_idx,x,y])
        # print("dLdA",dLdA)
        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        import pdb
        out_width = A.shape[2]-self.kernel + 1
        out_height = A.shape[3]-self.kernel + 1
        Z =  np.zeros((A.shape[0],A.shape[1],out_width,out_height))
        for batch in range(Z.shape[0]):
            for map_idx in range(Z.shape[1]):
                for x in range(out_height):
                    for y in range(out_width):
                        # pdb.set_trace()
                        Z[batch,map_idx,x,y] =np.mean(A[batch,map_idx,x:x+self.kernel,y:y+self.kernel] )
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros_like(self.A)
        for batch in range(dLdZ.shape[0]):
            for map_idx in range(dLdZ.shape[1]):
                for x in range(dLdZ.shape[2]):
                    for y in range(dLdZ.shape[3]):
                        for i in range (self.kernel):
                            for j in range(self.kernel):
                                dLdA[batch,map_idx,x+i,y+j] +=((1/(self.kernel*self.kernel))* dLdZ[batch,map_idx,x,y])

        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A =  self.meanpool2d_stride1.forward(A)
        Z =  self.downsample2d.forward(A)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        
        return dLdA
