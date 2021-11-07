import torch.nn as nn
from math import log, sqrt
from torch.nn import functional as F
from models.modules.activations import xUnitS
from models.modules.misc import UpsampleX2

__all__ = ['g_srgan', 'g_xsrgan']

def initialize_model(model, scale=1.):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            continue

class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False):
        super(BasicBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.lrelu(self.bn(self.conv(x)))
        return x

class xBasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False):
        super(xBasicBlock, self).__init__()
        self.xunit = xUnitS(num_features=out_channels, batch_norm=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.xunit(self.bn(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, num_features=64, kernel_size=3, padding=1, bias=False):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        residual = x
        x = self.conv2(self.relu(self.conv1(x)))
        x = residual + x
        return x

class xResBlock(nn.Module):
    def __init__(self, num_features=64, kernel_size=3, padding=1, bias=False):
        super(xResBlock, self).__init__()
        self.xunit = xUnitS(num_features=num_features)
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        residual = x
        x = self.conv2(self.xunit(self.conv1(x)))
        x = residual + x
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks, scale, block):
        super(Vanilla, self).__init__()
        # parameters
        self.scale_factor = scale

        # features
        blocks = [block(in_channels=in_channels, out_channels=num_features, kernel_size=5, padding=2)]
        for _ in range(1, num_blocks):
            blocks.append(block(in_channels=num_features, out_channels=num_features))
        self.features = nn.Sequential(*blocks)

        # features to image
        self.features_to_image = nn.Conv2d(in_channels=num_features, out_channels=in_channels, kernel_size=5, padding=2)
        
        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = F.interpolate(x, size=(self.scale_factor * x.size(-2), self.scale_factor * x.size(-1)), mode='bicubic', align_corners=True)
        r = x
        x = self.features(x)
        x = self.features_to_image(x)
        x += r
        return x

class SRGAN(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks, scale, block):
        super(SRGAN, self).__init__()
        # parameters
        self.scale = scale
        
        # image to features
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.image_to_features = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=5, padding=2)

        # features
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block(num_features=num_features))
        self.features = nn.Sequential(*blocks)

        # upsampling
        blocks = []
        for _ in range(int(log(scale, 2))):
            block = UpsampleX2(in_channels=num_features, out_channels=num_features)
            blocks.append(block)
        self.usample = nn.Sequential(*blocks)

        # features to image
        self.hrconv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.features_to_image = nn.Conv2d(in_channels=num_features, out_channels=in_channels, kernel_size=5, padding=2)

        # initialize weights
        initialize_model(self)

    def forward(self, x):
        r = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.lrelu(self.image_to_features(x))
        x = self.features(x)
        x = self.usample(x)
        x = self.features_to_image(self.lrelu(self.hrconv(x)))
        x = x + r
        return x

def g_vanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 8)
    config.setdefault('scale', 4)
    
    config['block'] = BasicBlock
    return Vanilla(**config)

def g_xvanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 8)
    config.setdefault('scale', 4)
    
    config['block'] = xBasicBlock
    return Vanilla(**config)

def g_srgan(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 16)
    config.setdefault('scale', 4)

    config['block'] = ResBlock
    return SRGAN(**config)

def g_xsrgan(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 10)
    config.setdefault('scale', 4)

    config['block'] = xResBlock
    return SRGAN(**config)