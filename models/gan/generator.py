import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim=100, im_shape=(1,28,28), inner_feature=256):
        '''
            input: N * z_dim
            output: N * channel * height * width
        '''
        super().__init__()
        self.im_shape = im_shape
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            nn.Linear(z_dim,inner_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(inner_feature, int(np.prod(im_shape))),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x).view(x.size(0), *self.im_shape)