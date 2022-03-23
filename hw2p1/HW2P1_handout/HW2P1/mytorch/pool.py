from re import S
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
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape

        out_channels = in_channels
        output_width = (input_width - self.kernel) + 1
        output_height = (input_height - self.kernel) + 1
        
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        
        for b in range(batch_size):
            for cout in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        sliced = A[b,cout,w:w+self.kernel,h:h+self.kernel]
                        Z[b,cout,w,h] = np.max(sliced)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        in_channels = out_channels
        input_width, input_height = (output_width - 1) + self.kernel, (output_height - 1) + self.kernel
        
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for cout in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        sliced = self.A[b,cout,w:w+self.kernel,h:h+self.kernel]
                        mask = (sliced == np.max(sliced))

                        dLdA[b,cout,w:w+self.kernel,h:h+self.kernel] += np.multiply(mask, dLdZ[b,cout,w,h])

        
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
        batch_size, in_channels, input_width, input_height = A.shape

        out_channels = in_channels
        output_width = (input_width - self.kernel) + 1
        output_height = (input_height - self.kernel) + 1
        
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        
        for b in range(batch_size):
            for cout in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        sliced = A[b,cout,w:w+self.kernel,h:h+self.kernel]
                        Z[b,cout,w,h] = np.average(sliced)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        in_channels = out_channels
        input_width, input_height = (output_width - 1) + self.kernel, (output_height - 1) + self.kernel
        
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for cout in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        v = dLdZ[b,cout,w,h]
                        avg = v / (self.kernel * self.kernel)
                        out = np.ones((self.kernel, self.kernel)) * avg

                        dLdA[b,cout,w:w+self.kernel,h:h+self.kernel] += out
        
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel=self.kernel) #TODO
        self.downsample2d = Downsample2d(downsampling_factor=stride) #TODO

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
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel=self.kernel) #TODO
        self.downsample2d = Downsample2d(downsampling_factor=stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA
