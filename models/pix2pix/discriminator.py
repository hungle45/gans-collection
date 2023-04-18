import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, im_shape=(3,256,256), features=[64,128,256,512]):
        '''
            input: (N, channel, height, width), (N, channel, height, width)
            output: (N, 1)
        '''
        super().__init__()
        im_channel = im_shape[0]

        layers = nn.ModuleList() 
        in_feature = features[0]
        for feature in features[1:]:
            layers.append(
                self._bacic_block(in_feature, feature, 4, 1 if feature == features[-1] else 2, 1)
            )
            in_feature = feature
        
        self.disc = nn.Sequential(
            nn.Conv2d(im_channel*2, features[0], 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            *layers,
            nn.Conv2d(in_feature, 1, 4, 1, 1, padding_mode='reflect'),
        )


    def _bacic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        return self.disc(x)