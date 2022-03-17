# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from torch import batch_norm
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
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
        batch_size, in_channel, input_size = A.shape
        output_size = (input_size - self.kernel_size) + 1

        Z = np.zeros([batch_size, self.out_channels, output_size])
        
        for batch in range(batch_size):
            for channel in range(self.out_channels):
                for idx in range(output_size):
                    Z[batch, channel, idx] = (A[batch, :, idx:idx+self.kernel_size] * self.W[channel,:,:]).sum()
                Z[batch,channel] += self.b[channel]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        
        for batch in range(batch_size):
            for cout in range(self.out_channels):
                for cin in range(self.in_channels):
                    for kernel in range(self.kernel_size):
                        for idx in range(output_size):
                            self.dLdW[cout,cin,kernel] += self.A[batch,cin,kernel+idx]*dLdZ[batch,cout,idx]
      
        self.dLdb = np.sum(dLdZ, axis=(0,2)) # TODO
        dLdA = np.zeros(self.A.shape) # TODO
        
        for batch in range(batch_size):
            for cout in range(self.out_channels):
                for cin in range(self.in_channels):
                    for kernel in range(self.kernel_size):
                        for idx in range((dLdA.shape[2] - self.kernel_size) + 1):
                            dLdA[batch,cin,idx+kernel] += self.W[cout,cin,kernel]*dLdZ[batch,cout,idx]
        
        
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn) # TODO
        self.downsample1d = Downsample1d(downsampling_factor=stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        A = self.conv1d_stride1.forward(A)
        Z = self.downsample1d.forward(A)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdA = self.downsample1d.backward(dLdZ)
        
        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

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
        batch_size, in_channel, input_width, input_height = A.shape
        output_width = (input_width - self.kernel_size) + 1
        output_height = (input_height - self.kernel_size) + 1
        
        #TODO
        Z = np.zeros([batch_size, self.out_channels, output_width, output_height])
            
        for batch in range(batch_size):
            for channel in range(self.out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        Z[batch, channel, w, h] = (A[batch, :, w:w+self.kernel_size, h:h+self.kernel_size] * self.W[channel,:,:,:]).sum()
                Z[batch,channel] += self.b[channel]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        _, in_channels, input_width, input_height = self.A.shape
        
        for b in range(batch_size):
            for cout in range(self.out_channels):
                for cin in range(self.in_channels):
                    for w in range(input_width-output_width+1):
                        for h in range(input_height-output_height+1):
                            self.dLdW[cout,cin,w,h] += (self.A[b,cin,w:w+output_width,h:h+output_height] * dLdZ[b,cout,:,:]).sum()
                        
                        
        self.dLdb = np.sum(np.sum(dLdZ, axis=(0,3)), axis=1) # TODO
        dLdA = np.zeros(self.A.shape)

        pad_dLdZ = np.pad(dLdZ,((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1)))

        for b in range(batch_size):
            for cin in range(self.in_channels):
                for cout in range(self.out_channels):
                    for w in range(input_width):
                        for h in range(input_height):
                            dLdA[b,cin,w,h] += (pad_dLdZ[b,cout,w:w+self.kernel_size,h:h+self.kernel_size] * np.flip(np.flip(self.W[cout][cin], axis=1), axis=0)).sum()

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn) # TODO
        self.downsample2d = Downsample2d(downsampling_factor=stride) # TODO

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
        Z = self.downsample2d.forward(A)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample2d backward
        # TODO
        dLdA = self.downsample2d.backward(dLdZ)
        
        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA) # TODO 

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor=upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor=upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A = A

        batch_size, _, _ = A.shape
        Z = A.reshape(batch_size,-1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        
        dLdA = dLdZ.reshape(self.A.shape) #TODO

        return dLdA

