import math

import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_classes, im_shape=(1,32,32), inner_feature=64):
        '''
            input: (N, channel, height, width), (N,)
            output: (N, 1, 1, 1)
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
            nn.Conv2d(im_channel+1, inner_feature, 4, 2, 1),
            nn.LeakyReLU(0.2),
            *inner_blocks,
            nn.Conv2d(inner_feature*2**n_inner_block, 1, 4, 2, 1),
            nn.Sigmoid()
        )

        self.embed = nn.Embedding(num_classes, np.prod(im_shape))

    def _basic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, im, label):
        embed_label = self.embed(label).view(label.shape[0], *self.im_shape)
        inp = torch.cat([im,embed_label],dim=1)
        return self.disc(inp)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disc = Discriminator(10).to(device)
    inp = torch.randn((2,1,32,32)).to(device)
    lab = torch.randn((2,1)).to(device).to(torch.int32)
    print(disc(inp,lab).shape)

if __name__ == '__main__':
    test()