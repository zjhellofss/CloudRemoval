import torch
import torch.nn as nn
from collections import OrderedDict
from models.layers import CBR
from models.models_utils import weights_init


class _Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, conv_dim=64, layer_num=4):
        super(_Discriminator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(3 + 3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(current_dim * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.model(x)
        out_src = self.conv_src(x)
        return out_src


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator())]))

        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)


