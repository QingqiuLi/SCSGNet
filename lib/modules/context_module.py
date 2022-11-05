import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


class SPFF_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(SPFF_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class SPFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPFF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = SPFF_kernel(in_channel, out_channel, 3)
        self.branch2 = SPFF_kernel(in_channel, out_channel, 5)
        self.branch3 = SPFF_kernel(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x01   = x0+x1
        x012  = x01+x2
        x0123 = x012+x3

        x_cat = self.conv_cat(torch.cat((x0, x01, x012, x0123), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x
