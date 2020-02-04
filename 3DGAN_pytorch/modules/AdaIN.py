import torch
import torch.nn as nn

from .InstanceNorm import InstanceNorm
from .PixelNorm import PixelNorm
from .FC import FC

class AdaIN(nn.Module):
    def __init__(self, dlatent_size, num_feature, use_noise, use_pixel_norm, use_instance_norm, use_wscale):
        super(AdaIN , self).__init__()

        if use_noise:
            self.noise = ApplyNoise(num_feature)
        else:
            self.noise = None
 
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        self.style_mode = ApplyStyle(dlatent_size, num_feature, use_wscale=use_wscale)

    def forward(self, x, noise, latent):
        if self.noise is not None:
            x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        x = self.style_mode(x, latent)
        return x


class ApplyNoise(nn.Module):
    def __init__(self, num_feature):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(num_feature))
        
    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device, dtype=x.dtype)

        x += self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)
        return x


class ApplyStyle(nn.Module):
    def __init__(self, dlatent_size, num_feature, use_wscale):
        super(ApplyStyle, self).__init__()
        self.num_feature = num_feature
        self.linear = FC(dlatent_size, 2*num_feature, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent) # B x channels*2
        shape = [-1, 2, x.size(1), 1, 1, 1]
        style = style.view(shape)  # B x 2 x channels x 1 x 1 x 1
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x
