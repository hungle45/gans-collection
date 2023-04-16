import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_classes, embed_size, 
                 im_shape=(1,28,28), inner_feature=128):
        '''
            input: N * channel * height * width
            output: N * 1
        '''
        super().__init__()
        self.disc = nn.Sequential(
            self._bacic_block(np.prod(im_shape)+embed_size, inner_feature),
            self._bacic_block(inner_feature, inner_feature*2),
            self._bacic_block(inner_feature*2, inner_feature*4),
            self._bacic_block(inner_feature*4, inner_feature*8),
            nn.Linear(inner_feature*8, 1),
            nn.Sigmoid()
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _bacic_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2)
        )

    def forward(self, im, label):
        im = im.view(im.size(0),-1)
        embed_label = self.embed(label)
        inp = torch.cat([im,embed_label],dim=1)
        return self.disc(inp)