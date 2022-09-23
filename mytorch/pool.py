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
        input_width = A.shape[2]
        input_height = A.shape[3]
        output_width = (input_width - self.kernel)//1 + 1
        output_height = (input_height - self.kernel)//1 + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        
        self.index_map = np.zeros((A.shape[0], A.shape[1],output_width, output_height, 2))
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for x in range(output_width):
                    for y in range(output_height):
                        Z[i, j, x, y] = np.max(A[i, j, x: x + self.kernel, y: y + self.kernel])
                        temp = np.unravel_index(np.argmax(A[i, j, x: x + self.kernel, y: y + self.kernel]), (self.kernel , self.kernel))
                        self.index_map[i, j, x, y, 0] = int(x + temp[0])
                        self.index_map[i, j, x, y, 1] = int(y + temp[1])
                        #print(np.argmax(A[i, j, x: x + self.kernel, y: y + self.kernel]))
                        
        
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        output_width = dLdZ.shape[2]
        output_height = dLdZ.shape[3]
        
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.A.shape[2], self.A.shape[3]))
        
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for x in range(output_width):
                    for y in range(output_height):
                            dLdA[i, j, int(self.index_map[i, j, x, y, 0]), int(self.index_map[i, j, x, y, 1])] +=  dLdZ[i, j, x, y]
        

        
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
        input_width = A.shape[2]
        input_height = A.shape[3]
        output_width = (input_width - self.kernel)//1 + 1
        output_height = (input_height - self.kernel)//1 + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        
        #self.index_map = np.zeros((A.shape[0], A.shape[1],output_width, output_height, 2))
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for x in range(output_width):
                    for y in range(output_height):
                        Z[i, j, x, y] = np.mean(A[i, j, x: x + self.kernel, y: y + self.kernel])
                        #temp = np.unravel_index(np.argmax(A[i, j, x: x + self.kernel, y: y + self.kernel]), (self.kernel , self.kernel))
                        #self.index_map[i, j, x, y, 0] = int(x + temp[0])
                        #self.index_map[i, j, x, y, 1] = int(y + temp[1])
                        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        output_width = dLdZ.shape[2]
        output_height = dLdZ.shape[3]
        
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.A.shape[2], self.A.shape[3]))
        
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for x in range(output_width):
                    for y in range(output_height):
                        derv = dLdZ[i, j, x, y] / (self.kernel * self.kernel)
                        for l in range(self.kernel):
                            for k in range(self.kernel):
                                dLdA[i, j, x + l, y + k] +=  derv
                                
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        inter = self.maxpool2d_stride1.forward(A)
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
        inter = self.downsample2d.backward(dLdZ)
        
        dLdA = self.maxpool2d_stride1.backward(inter)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        inter = self.meanpool2d_stride1.forward(A)
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
        inter = self.downsample2d.backward(dLdZ)
        
        dLdA = self.meanpool2d_stride1.backward(inter)

        return dLdA