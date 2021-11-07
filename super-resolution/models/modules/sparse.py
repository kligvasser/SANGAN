import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparse(object):
    def __init__(self, gain=1., update_rate=2, eps=1e-8, momentum=0.):
        self.steps = 0
        self.gain = gain
        self.update_rate = update_rate
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('s', torch.ones(1))

    # Compute the spectrally-normalized weight
    def W_(self, input_size):
        if self.training and (self.steps % self.update_rate == 0):
            with torch.no_grad():
                w_fft = F.pad(self.weight, (0, input_size[2] - self.weight.size(2), 0, input_size[3] - self.weight.size(3)))
                # w_fft = torch.rfft(w_fft, signal_ndim=2, onesided=False) # for pytorch version < 1.7
                w_fft = torch.fft.fft2(w_fft)
                r = torch.norm(w_fft, p=2, dim=-1)
                self.s = self.s * self.momentum + (1. - self.momentum) * (r.max() * self.gain + self.eps)
        self.steps += 1
        return self.weight / self.s

class ApproxSparse(object):
    def __init__(self, gain=1.0, update_rate=2, eps=1e-8, momentum=0.8, rate=0.5):
        self.steps = 0
        self.gain = gain
        self.update_rate = update_rate
        self.eps = eps
        self.momentum = momentum
        self.rate = rate
        self.register_buffer('s', torch.ones(1))

    # Compute the spectrally-normalized weight
    def W_(self, input_size):
        if self.training and (self.steps % self.update_rate == 0):
            with torch.no_grad():
                w = self.weight[torch.randperm(self.weight.size(0))[:int((self.weight.size(0) * self.rate))], :, :, :]
                w = w[:, torch.randperm(self.weight.size(1))[:int((self.weight.size(1) * self.rate))], :, :]
                w_fft = F.pad(w, (0, input_size[2] - self.weight.size(2), 0, input_size[3] - self.weight.size(3)))
                # w_fft = torch.rfft(w_fft, signal_ndim=2, onesided=False) # for pytorch version < 1.7
                w_fft = torch.fft.fft2(w_fft)
                r = torch.norm(w_fft, p=2, dim=-1)
                self.s = self.s * self.momentum + (1. - self.momentum) * (r.max() * self.gain + self.eps)
        self.steps += 1
        return self.weight / self.s

class SparseCirConv2d(nn.Conv2d, Sparse):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True,
                 gain=1., update_rate=2, eps=1e-8, momentum=0.):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        Sparse.__init__(self, gain=gain, update_rate=update_rate, eps=eps, momentum=momentum)
        self.p2d = (padding, padding, padding, padding)

    def forward(self, x):
        x_size = x.size()
        x = F.pad(x, pad=self.p2d, mode='circular')
        return F.conv2d(x, self.W_(x_size), self.bias, self.stride, 0, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', gain={gain}, update_rate={update_rate}, eps={eps}, momentum={momentum}'
        return s.format(**self.__dict__)

class ApproxSparseCirConv2d(nn.Conv2d, ApproxSparse):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True,
                 gain=1.2, update_rate=1, eps=1e-8, momentum=0.9, rate=0.5):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        ApproxSparse.__init__(self, gain=gain, update_rate=update_rate, eps=eps, momentum=momentum, rate=rate)
        self.p2d = (padding, padding, padding, padding)

    def forward(self, x):
        x_size = x.size()
        x = F.pad(x, pad=self.p2d, mode='circular')
        return F.conv2d(x, self.W_(x_size), self.bias, self.stride, 0, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', gain={gain}, update_rate={update_rate}, eps={eps}, momentum={momentum}, rate={rate}'
        return s.format(**self.__dict__)