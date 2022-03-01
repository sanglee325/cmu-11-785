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
        k = self.upsampling_factor
        output_width = A.shape[2] * k - (k - 1)
        
        size = (A.shape[0], A.shape[1], output_width)
        Z = np.zeros(size)
        up_idx = np.arange(0, output_width, k)
        Z[:,:,up_idx] = A
        
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.upsampling_factor
        
        down_idx = np.arange(0, dLdZ.shape[2], k)
        dLdA = dLdZ[:,:,down_idx]  #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.orig_size = None

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.orig_size = A.shape
        k = self.downsampling_factor
        
        down_idx = np.arange(0, A.shape[2], k)
        Z = A[:,:,down_idx]  #TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.downsampling_factor
        
        size = (dLdZ.shape[0], dLdZ.shape[1], self.orig_size[2])
        dLdA = np.zeros(size)
        up_idx = np.arange(0, self.orig_size[2], k)
        dLdA[:,:,up_idx] = dLdZ

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
        k = self.upsampling_factor
        w_size_dilated = A.shape[2] * k - (k - 1)
        h_size_dilated = A.shape[3] * k - (k - 1)
        
        size = (A.shape[0], A.shape[1], w_size_dilated, h_size_dilated)
        Z = np.zeros(size)

        Z[:,:,::k,::k] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        k = self.upsampling_factor
        
        dLdA = dLdZ[:,:,::k,::k]  #TODO

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.orig_size = None

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        self.orig_size = A.shape
        k = self.downsampling_factor
        
        Z = A[:,:,::k,::k]  #TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        k = self.downsampling_factor
        
        size = self.orig_size
        dLdA = np.zeros(size)
        
        dLdA[:,:,::k,::k] = dLdZ

        return dLdA