import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, num_classes, embed_size, z_dim=100, 
                 im_shape=(1,28,28), inner_feature=128):
        '''
            input: (N, z_dim), (N,)
            output: (N, channel, height, width)
        '''
        super().__init__()
        self.im_shape = im_shape
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self._bacic_block(z_dim+embed_size, inner_feature),
            self._bacic_block(inner_feature, inner_feature*2),
            self._bacic_block(inner_feature*2, inner_feature*4),
            self._bacic_block(inner_feature*4, inner_feature*8),
            nn.Linear(inner_feature*8, int(np.prod(im_shape))),
            nn.Sigmoid()
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _bacic_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, label):
        embed_label = self.embed(label)
        inp = torch.cat([x,embed_label],dim=1)
        return self.gen(inp).view(x.size(0), *self.im_shape)