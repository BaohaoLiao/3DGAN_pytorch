import torch
import torch.nn as nn
import numpy as np
import math

from . import (
    GANGenerator, register_generator, register_generator_architecture,
)
from modules.FC import FC
from modules.PixelNorm import PixelNorm
from modules.InstanceNorm import InstanceNorm
from modules.Conv3d import Conv3d
from modules.RRDBNet import RRDBNet
from modules.AdaIN import AdaIN



@register_generator('stylegan_generator')
class StyleGANGenerator(GANGenerator):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        input: a batch of low resolution images
    """
    def __init__(self, args):
        super(StyleGANGenerator, self).__init__()
        self.mapping_fmaps = args.mapping_fmaps
        self.hidden_size = args.hidden_size
        self.dlatent_size = args.dlatent_size
        self.style_mixing_prob = args.style_mixing_prob
        self.truncation_psi = args.truncation_psi
        self.truncation_cutoff = args.truncation_cutoff
        self.num_layers = args.num_res * 2
        self.num_res = args.num_res
        self.known_filter = args.known_filter

        features = args.G_num_features.split(':')
        assert len(features) == args.num_res, 'number of generator features does not match number of resolution'
        self.features = list(map(int, features))

        self.mapping = G_mapping(self.mapping_fmaps, self.hidden_size, self.dlatent_size, self.known_filter)
        self.projection = G_projection(in_nc=args.num_channels, out_nc=args.projection_output_channels,
                                       nf=args.RRDB_feature, nb=args.RRDB_blocks)

        self.conv = []
        for n in range(args.num_res):
            self.conv.append(nn.Conv3d(args.projection_output_channels, self.features[n], kernel_size=3, padding=1))
        self.conv = nn.ModuleList(self.conv)

        self.downsample = nn.AvgPool3d(kernel_size=4, stride=2, padding=1)

        self.synthesis = G_synthesis(args.img_size, args.num_channels, self.features, self.dlatent_size)
        self.tanh = nn.Tanh()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--style-mixing-prob', type=float, default=0.9, metavar='S')
        parser.add_argument('--truncation-psi', type=float, default=0.7, metavar='T')
        parser.add_argument('--truncation-cutoff', type=int, default=4, metavar='T')
        parser.add_argument('--dlatent-size', type=int, metavar='L')
        parser.add_argument('--hidden-size', type=int, metavar='L')
        parser.add_argument('--num-res', type=int, default=4, metavar='N')
        parser.add_argument('--G-num-features', type=str, default="64:64:128:128", metavar='N')
        parser.add_argument('--projection-output-channels', type=int, default=64, metavar='P')
        parser.add_argument('--RRDB-feature', type=int, default=64, metavar='R')
        parser.add_argument('--RRDB-blocks', type=int, default=3, metavar='R')
        parser.add_argument('--use-noise', action='store_false')
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        return StyleGANGenerator(args)

    def forward(self, latents1, lr):
        dlatents1 = self.mapping(latents1)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, self.num_layers, -1)  # B x num_layer x latent_size

        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, self.num_layers, 1], dtype=np.float32)
            for i in range(self.num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        projection = self.projection(lr)
        projections = []
        projections.append(projection)
        for i in range(self.num_res - 1):
            projections.insert(0, self.downsample(projections[0]))

        resolutions = []
        for i in range(self.num_res):
            resolutions.append(self.conv[i](projections[i]))

        img = self.synthesis(dlatents1, resolutions)
        img = self.tanh(img)
        return img


class G_projection(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(G_projection, self).__init__()
        self.projection = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)

    def forward(self, lr):
        x = self.projection(lr)
        return x


class G_mapping(nn.Module):
    def __init__(self, mapping_fmaps, hidden_size, dlatent_size, known_filter=False, normalize_latents=True,
                 use_wscale=True, lrmul=0.01, gain=2**(0.5)):
        super(G_mapping, self).__init__()
        self.known_filter = known_filter
        self.embed_scale = math.sqrt(mapping_fmaps)
        if self.known_filter:
            self.emb = Embedding(8, mapping_fmaps)

        self.func = nn.Sequential(
            FC(mapping_fmaps, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, hidden_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(hidden_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )
        self.normalize_latents = normalize_latents
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.known_filter:
            x = self.embed_scale * self.emb(x)

        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out


class G_synthesis(nn.Module):
    def __init__(self, img_size, num_channels, num_features, dlatent_size, use_pixel_norm=False, use_instance_norm=True,
                 use_noise=True, use_wscale = True, num_res=4):
        super(G_synthesis, self).__init__()
        assert use_instance_norm != use_pixel_norm, 'can only use instance norm or pixel norm'
        self.num_res = num_res

        # First block
        self.conv1 = Conv3d(num_features[0], num_features[0], kernel_size=3, use_wscale=use_wscale)
        self.adaIn1 = AdaIN(dlatent_size, num_features[0], use_noise, use_pixel_norm, use_instance_norm, use_wscale)
        self.conv2 = Conv3d(num_features[0], num_features[0], kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = AdaIN(dlatent_size, num_features[0], use_noise, use_pixel_norm, use_instance_norm, use_wscale)

        # generate noise
        # img_size = 128
        sizes = []
        for r in range(num_res-1, -1, -1):
            sizes.append(img_size // 2**(r))

        self.noise = []
        for r in range(num_res):
            temp = []
            shape = [1, 1, sizes[r], sizes[r], sizes[r]]
            for _ in range(2):
                temp.append(torch.randn(*shape).to("cuda"))
                #temp.append(torch.zeros(*shape).to("cuda"))
            self.noise.append(temp)

        # to original number of channels
        self.channel_shrinkage = Conv3d(num_features[-1], num_features[-1] // 2, kernel_size=3, use_wscale=use_wscale)
        self.torgb = Conv3d(num_features[-1] // 2, num_channels, kernel_size=3, gain=1, use_wscale=use_wscale)

        # generator blocks
        self.GBlocks = []
        for n in range(num_res - 1):
            self.GBlocks.append(
                GBlock(num_features[n], num_features[n + 1], self.noise[n + 1], dlatent_size, use_noise, use_pixel_norm,
                       use_instance_norm, use_wscale)
            )
        self.GBlocks = nn.ModuleList(self.GBlocks)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, dlatents, resolutions):
        x = resolutions[0]
        x = self.conv1(x)
        x = self.adaIn1(x, self.noise[0][0], dlatents[:, 0, :])
        x = self.conv2(x)
        x = self.adaIn2(x, self.noise[0][1], dlatents[:, 1, :])
        for n in range(self.num_res - 1):
            x = self.GBlocks[n](x, dlatents[:, 2*(n+1):2*(n+2), :], resolutions[n+1])
        x = self.act(self.channel_shrinkage(x))
        x = self.torgb(x)
        return x


class GBlock(nn.Module):
    def __init__(self, previous_num_feature, current_num_feature, noise_input, dlatent_size, use_noise, use_pixel_norm,
                 use_instance_norm, use_wscale, num_layers=2):
        super(GBlock, self).__init__()
        self.upsample = nn.ConvTranspose3d(previous_num_feature, current_num_feature, 4, stride=2, padding=1)
        self.num_layers = num_layers
        self.noise_input = noise_input

        self.adaIn = []
        self.conv = []
        for _ in range(num_layers):
            self.adaIn.append(AdaIN(dlatent_size, current_num_feature, use_noise, use_pixel_norm, use_instance_norm,
                                    use_wscale))
            self.conv.append(Conv3d(current_num_feature, current_num_feature, kernel_size=3, use_wscale=use_wscale))
        self.adaIn = nn.ModuleList(self.adaIn)
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x, latent, resolution):
        x = self.upsample(x)
        x = x + resolution
        for n in range(self.num_layers):
            x = self.conv[n](x)
            x = self.adaIn[n](x, self.noise_input[n], latent[:, n, :])
        return x


class LayerEpilogue(nn.Module):
    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
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
            self.instance.norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatent_in_slice=None):
        if self.noise is not None:
            x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatent_in_slice)

        return x


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size, channels*2, gain =1.0, use_wscale=use_wscale)
 
    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device,
                                dtype=x.dtype)
        return x + self.weight.view(1,-1,1,1,1) * noise.to(x.device)

def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

@register_generator_architecture('stylegan_generator', 'stylegan_generator')
def base_architecture(args):
    return

  
