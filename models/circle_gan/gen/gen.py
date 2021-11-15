'''
Define Models: Discriminator and Generator
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from models.models_utils import weights_init, print_network



class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, stride=True):
        super(UNetDown, self).__init__()

        layers = []
        if stride:
            layers.append(nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            layers.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        self.in_size = in_size
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True),
                  ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.output(x)
        return x


class _Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(_Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 1024, dropout=0.5)

        self.up1 = UNetUp(1024, 512, dropout=0.5)
        self.up2 = UNetUp(512, 256, dropout=0.5)
        self.up3 = UNetUp(256, 128, dropout=0.5)
        self.up4 = UNetUp(128, 64, dropout=0.5)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        output = self.final(u4)
        return output


class Generator(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.gen = nn.Sequential(OrderedDict([('gen', _Generator())]))

        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)
