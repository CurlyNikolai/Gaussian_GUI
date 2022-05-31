import math
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
import numpy as np 

def gaussian_rot(x, y, sigma_u, sigma_v, theta=0.0):
    return math.exp(- 0.5 * ((x * math.cos(theta) + y * math.sin(theta))**2 / (sigma_u*sigma_u) + (-x * math.sin(theta) + y * math.cos(theta))**2 / (sigma_v*sigma_v)))


class GaussianFilter(nn.Module):

    def __init__(self, channels, kernel_size, sigma_x, sigma_y, theta=0.0):
        super(GaussianFilter, self).__init__()

        # Create gaussian kernel image
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = gaussian_rot(
                    i-kernel_size//2, j-kernel_size//2, sigma_x, sigma_y, theta)

        # Normalize kernel
        kernel /= kernel.sum()

        # Transform kernel from nupy array to pytorch tensor, restructure for conv2d and send to cuda
        transform = transforms.ToTensor()
        kernel = transform(kernel.copy())
        kernel = kernel.view(1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim()-1))
        kernel = kernel.float()
        kernel = kernel.to(torch.device('cuda'))

        # Set the weight and group of the filter
        self.weight = kernel
        self.groups = channels
        
        self.conv = F.conv2d

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)
