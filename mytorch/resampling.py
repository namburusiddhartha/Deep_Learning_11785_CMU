import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)0
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        isize = A.shape[2]
        upsampledsize =  isize * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((A.shape[0], A.shape[1], upsampledsize))
        index = 0
        for i in range(isize):
            Z[:, :, index] = A[:, :, i]
            index = index + (self.upsampling_factor)
            
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        usize = dLdZ.shape[2]
        isize =  int(np.floor((usize + (self.upsampling_factor - 1))/ self.upsampling_factor))
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], isize))
        index = 0
        for i in range(isize):
            dLdA[:, :, i] = dLdZ[:, :, index]
            index = index + (self.upsampling_factor)
        

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

        usize = A.shape[2]
        self.usize = usize
        isize =  int(np.floor((usize + (self.downsampling_factor - 1))/ self.downsampling_factor)) 
        Z = np.zeros((A.shape[0], A.shape[1], isize))
        index = 0
        for i in range(isize):
            Z[:, :, i] = A[:, :, index]
            index = index + self.downsampling_factor
        

        return Z


    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        isize = dLdZ.shape[2]
        #if self.usize %2 == 0:
        usize =  self.usize
        #else:
            #usize =  isize * self.downsampling_factor - (self.downsampling_factor - 1)
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], usize))
        index = 0
        for i in range(isize):
            dLdA[:, :, index] = dLdZ[:, :, i]
            index = index + self.downsampling_factor
            
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

        input_width = A.shape[2]
        output_width =  input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        input_height = A.shape[3]
        output_height =  input_height * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        indexi = 0
        for i in range(input_width):            
            indexj = 0
            for j in range(input_height):
                Z[:, :, indexi, indexj] = A[:, :, i, j]
                indexj = indexj + (self.upsampling_factor)
            indexi = indexi + (self.upsampling_factor)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        input_width = dLdZ.shape[2]
        output_width =  int(np.floor((input_width + (self.upsampling_factor - 1))/ self.upsampling_factor))
        input_height = dLdZ.shape[3]
        output_height =  int(np.floor((input_height + (self.upsampling_factor - 1))/ self.upsampling_factor))
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], output_width, output_height))
        indexi = 0
        for i in range(output_width):      
            indexj = 0
            for j in range(output_height):
                dLdA[:, :, i, j] = dLdZ[:, :, indexi, indexj]
                indexj = indexj + (self.upsampling_factor)
            indexi = indexi + (self.upsampling_factor)

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

        input_width = A.shape[2]
        self.input_width = input_width
        output_width =  int(np.floor((input_width + (self.downsampling_factor - 1))/ self.downsampling_factor)) 
        input_height = A.shape[3]
        self.input_height = input_height
        output_height =  int(np.floor((input_height + (self.downsampling_factor - 1))/ self.downsampling_factor)) 
        
        
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        indexi = 0
        for i in range(output_width):      
            indexj = 0
            for j in range(output_height):
                Z[:, :, i, j] = A[:, :, indexi, indexj]
                indexj = indexj + (self.downsampling_factor)
            indexi = indexi + (self.downsampling_factor)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        input_width = dLdZ.shape[2]
        output_width =  self.input_width
        input_height = dLdZ.shape[3]
        output_height =  self.input_height
        
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], output_width, output_height))
        indexi = 0
        for i in range(input_width):      
            indexj = 0
            for j in range(input_height):
                dLdA[:, :, indexi, indexj] = dLdZ[:, :, i, j]
                indexj = indexj + (self.downsampling_factor)
            indexi = indexi + (self.downsampling_factor)
        
        
        

        return dLdA