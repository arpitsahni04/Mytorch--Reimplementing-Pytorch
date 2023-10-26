import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        W_out = self.upsampling_factor*(A.shape[2]-1)+1
        Z = np.zeros((A.shape[0], A.shape[1],W_out))  # TODO
        Z[:,:, ::self.upsampling_factor] = A[:] 
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        W_in = (dLdZ.shape[2]-1)/self.upsampling_factor + 1
        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],int(W_in)))  # TODO
        dLdA = dLdZ[:,:,::self.upsampling_factor]
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.W_in  =A.shape[2]
        W_out = A.shape[2]//self.downsampling_factor + 1
        Z = np.zeros((A.shape[0],A.shape[1],W_out))  # TODO
        Z = A[:,:,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1],self.W_in))  # TODO
        dLdA[:,:, ::self.downsampling_factor] = dLdZ[:] 
        
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        H_out = self.upsampling_factor*(A.shape[3]-1)+1
        W_out = self.upsampling_factor*(A.shape[2]-1)+1
        Z = np.zeros((A.shape[0],A.shape[1], W_out,H_out))  # TODO
        print("A",A.shape)
        print("Z",Z.shape)
        Z[:,:,::self.upsampling_factor, ::self.upsampling_factor] = A[:] 
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        H_in = (dLdZ.shape[3]-1)/self.upsampling_factor + 1
        W_in = (dLdZ.shape[2]-1)/self.upsampling_factor + 1
        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],int(W_in),int(H_in)))  # TODO
        dLdA = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        self.W_in  =A.shape[2]
        self.H_in  =A.shape[3]
        W_out = A.shape[2]//self.downsampling_factor + 1
        H_out = A.shape[3]//self.downsampling_factor + 1
        Z = np.zeros((A.shape[0],A.shape[1],W_out,H_out))  # TODO
        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1],self.W_in,self.H_in))  # TODO
        dLdA[:,:,::self.downsampling_factor, ::self.downsampling_factor] = dLdZ[:] 
        
        return dLdA
