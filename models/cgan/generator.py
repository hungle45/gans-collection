import math

import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, num_classes, embed_size, z_dim=100, 
                 im_shape=(1,32,32), inner_feature=128):
        '''
            input: (N, z_dim), (N,)
            output: (N, channel, height, width)
        '''
        super().__init__()
        if im_shape[-1] != im_shape[-2]:
            raise Exception('The image\'s width must equal its height.')

        im_channel = im_shape[0]
        im_size = im_shape[-1]
        self.im_shape = im_shape
        self.z_dim = z_dim

        n_inner_block = int(math.log2(im_size)) - 2
        if 2 ** (n_inner_block+2) != im_size:
            raise Exception('The image\'s size must be a power of two.')

        inner_blocks = []
        for i in range(n_inner_block):
            inner_blocks.append(
                self._basic_block(
                    inner_feature*(2**(n_inner_block-i)), 
                    inner_feature*(2**(n_inner_block-i-1)),
                    4, 2, 1
                ))

        self.gen = nn.Sequential(
            self._basic_block(z_dim+embed_size, inner_feature*(2**n_inner_block), 4, 2, 1),
            *inner_blocks,
            nn.ConvTranspose2d(inner_feature, im_shape[0], 4, 2, 1),
            nn.Tanh()
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _basic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, label):
        embed_label = self.embed(label)
        inp = torch.cat([x,embed_label],dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.gen(inp)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(10, 100).to(device)
    inp = torch.randn((2,100)).to(device)
    lab = torch.randn((2,)).to(device).to(torch.int32)
    print(gen(inp,lab).shape)

if __name__ == '__main__':
    test()