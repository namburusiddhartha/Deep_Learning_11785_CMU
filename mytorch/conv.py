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
        input_size = A.shape[2]
        output_size = (input_size - self.kernel_size)//1 + 1
        Z = np.zeros((A.shape[0], self.W.shape[0], output_size))
        
        for i in range(A.shape[0]):
            for x in range(output_size):
                for j in range(self.out_channels):
                    Z[i, j, x] = np.sum(A[i, :, x: x  + self.kernel_size] *  self.W[j, :, :]) + self.b[j] 
            
                        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = np.zeros((self.W.shape[0], self.W.shape[1], self.W.shape[2]))
                
        dLdA = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2]))
        flippedW = np.flip(self.W, axis = 2)
        
        pad = self.kernel_size - 1
        pd_dLdZ = np.pad(dLdZ, ((0,0), (0,0), (pad,pad)), 'constant', constant_values = 0)
        
        for i in range(self.A.shape[0]):
            for x in range(self.A.shape[2]):
                for j in range(self.in_channels):
                    dLdA[i, j, x] = np.sum(pd_dLdZ[i, :, x: x  + self.kernel_size] *  flippedW[:, j, :])
                    
        
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                for x in range(self.kernel_size):
                    self.dLdW[j, i, x] = np.sum(self.A[:, i, x: x + dLdZ.shape[2]] * dLdZ[:, j, :])
        
        
        self.dLdb = np.zeros(self.out_channels)
        for i in range(self.out_channels):
            self.dLdb[i] = np.sum(dLdZ[:, i, :])
        

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        inter = self.conv1d_stride1.forward(A)
        # downsample
        Z = self.downsample1d.forward(inter)

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
        inter = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(inter)

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
        input_width = A.shape[2]
        input_height = A.shape[3]
        output_width = (input_width - self.kernel_size)//1 + 1
        output_height = (input_height - self.kernel_size)//1 + 1
        Z = np.zeros((A.shape[0], self.W.shape[0], output_width, output_height))
        
        for i in range(A.shape[0]):
            for x in range(output_width):
                for y in range(output_height):
                    for j in range(self.out_channels):
                        Z[i, j, x, y] = np.sum(A[i, :, x: x  + self.kernel_size, y: y + self.kernel_size] *  self.W[j, :, :, :]) + self.b[j] 


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]))
        pad = self.kernel_size - 1
        pd_dLdZ = np.pad(dLdZ, ((0,0), (0,0), (pad,pad), (pad, pad)), 'constant', constant_values = 0)
        flippedW = np.flip(self.W, axis = 2)
        flippedW = np.flip(flippedW, axis = 3)

        for i in range(self.A.shape[0]):
            for x in range(self.A.shape[2]):
                for y in range(self.A.shape[3]):
                    for j in range(self.in_channels):
                        dLdA[i, j, x, y] = np.sum(pd_dLdZ[i, :, x: x  + self.kernel_size, y: y  + self.kernel_size] *  flippedW[:, j, :, :])
                        
        
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                for x in range(self.kernel_size):
                    for y in range(self.kernel_size):
                        self.dLdW[j, i, x, y] = np.sum(self.A[:, i, x: x + dLdZ.shape[2], y: y + dLdZ.shape[3]] * dLdZ[:, j, :, :])
                    
                    
        for i in range(self.out_channels):
            self.dLdb[i] = np.sum(dLdZ[:, i, :, :])

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        inter = self.conv2d_stride1.forward(A)
        # downsample
        Z = self.downsample2d.forward(inter)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        inter = self.downsample2d.backward(dLdZ)
        
        dLdA = self.conv2d_stride1.backward(inter)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)

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
        delta_out = self.conv1d_stride1.backward(dLdZ)

        dLdA =  self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.upsample2d = Upsample2d(upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)

        dLdA =  self.upsample2d.backward(delta_out)

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
        Z = np.reshape(A , (self.A.shape[0], self.A.shape[1] * self.A.shape[2]))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, (self.A.shape[0], self.A.shape[1], self.A.shape[2]))

        return dLdA

