import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, im_shape=(1,28,28), inner_feature=256):
        '''
            input: (N, channel, height, width)
            output: (N, 1)
        '''
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(np.prod(im_shape), inner_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(inner_feature, 1),
            nn.Sigmoid()
        )

    def forward(self, im):
        return self.disc(im.view(im.size(0), -1))