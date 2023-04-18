import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, im_shape=(3,256,256), features=[64, 128, 256, 512, 512, 512, 512]):
        '''
            input: (N, channel, height, width)
            output: (N, channel, height, width)
        '''
        super().__init__()
        im_channel = im_shape[0]

        self.initial = nn.Sequential(
            nn.Conv2d(im_channel, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.encoder = nn.ModuleList()
        in_feature = features[0]
        for feature in features[1:]:
            self.encoder.append(
                self._encoder_block(in_feature, feature, 4, 2, 1)
            )
            in_feature = feature
        

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, 4, 2, 1),
            nn.ReLU()
        )
        
        self.decoder = nn.ModuleList()
        for i, feature in enumerate(reversed(features)):
            self.decoder.append(
                self._decoder_block(in_feature*2 if i != 0 else in_feature, feature, 4, 2, 1, i < 3)
            )
            in_feature = feature
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_feature*2, im_channel, 4, 2, 1),
            nn.Tanh(),
        )

    def _encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _decoder_block(self, in_channels, out_channels, kernel_size, stride, padding, use_dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_init = x = self.initial(x)

        x_skip = []
        for layer in self.encoder:
            x = layer(x)
            x_skip.append(x)
        x_skip = x_skip[::-1]
        
        x = self.bottleneck(x)
        
        x = self.decoder[0](x)
        for i, layer in enumerate(self.decoder[1:]):
            x = layer(torch.cat([x, x_skip[i]], 1))

        return self.final(torch.cat([x, x_init], 1))