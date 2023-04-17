import math

import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, im_shape=(1,32,32), inner_feature=128):
        '''
            input: (N, channel, height, width)
            output: (N, 1)
        '''
        super().__init__()
        if im_shape[-1] != im_shape[-2]:
            raise Exception('The image\'s width must equal its height.')

        im_channel = im_shape[0]
        im_size = im_shape[-1]
        self.im_shape = im_shape

        n_inner_block = int(math.log2(im_size)) - 2
        if 2 ** (n_inner_block+2) != im_size:
            raise Exception('The image\'s size must be a power of two.')

        inner_blocks = []
        for i in range(n_inner_block):
            inner_blocks.append(
                self._basic_block(
                    inner_feature*(2**i), 
                    inner_feature*(2**(i+1)),
                    4, 2, 1
                ))

        self.disc = nn.Sequential(
            nn.Conv2d(im_channel, inner_feature, 4, 2, 1),
            nn.LeakyReLU(0.1),
            *inner_blocks,
            nn.Conv2d(inner_feature*2**n_inner_block, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def _basic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, im):
        return self.disc(im)