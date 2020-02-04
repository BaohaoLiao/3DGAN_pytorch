import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.AdaIN import AdaIN
from . import (
    GANGenerator, register_generator, register_generator_architecture,
)


@register_generator('stylegan_generator_noflatten')
class StyleGANNoFlattenGenerator(GANGenerator):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        input: a batch of low resolution images
    """
    def __init__(self, args):
        super().__init__()
        self.num_res = args.num_res
        self.img_size = args.img_size
        assert np.log2(self.img_size) - self.num_res >= 1, 'smallest resolution is 4'
        
        features = args.num_features.split(':')
        assert len(features) == self.num_res, 'number of features doesn not match number of resolution'
        self.features = list(map(int, features))

        num_layers_each_block = args.num_layers_each_block.split(':')
        assert len(num_layers_each_block) == self.num_res, 'number of blocks doesn not match number of resolution'
        self.num_layers_each_block = list(map(int, num_layers_each_block))

        self.use_noise = args.use_noise
        self.use_pixel_norm = args.use_pixel_norm
        self.use_instance_norm = args.use_instance_norm

        self.mapping = G_mapping(self.num_res)
        self.synthesis = G_synthesis(self.img_size, self.num_res, args.channels, self.features, 
                             self.num_layers_each_block, self.use_pixel_norm, self.use_instance_norm, self.use_noise)
        self.tanh = nn.Tanh()
       
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--num-res', type=int, metavar='N',
                            help='how many resolution sizes')
        parser.add_argument('--num-features', type=str, metavar='N',
                            help='how many features for each resolution size')
        parser.add_argument('--num-layers-each-block', type=str, metavar='N',
                            help='how many layers in each block')
        parser.add_argument('--use-noise', type=bool, default=False, metavar='U',
                            help='whether use noise, shut down for test')
        parser.add_argument('--use-pixel-norm', type=bool, default=False, metavar='U',
                            help='use pixel norm in adaptive instance normalization')
        parser.add_argument('--use-instance-norm', type=bool, default=True, metavar='U',
                            help='use instance norm in adaptive instance normalization')
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        return StyleGANNoFlattenGenerator(args)

    def forward(self, x):
        """
        Args:
            x [b, c, s, s, s]: a batch of filtered data
        """
        latents = self.mapping(x)
        x = self.synthesis(latents)
        img = self.tanh(x)
        return img
        
        
class G_synthesis(nn.Module):
    def __init__(self, img_size, num_res, num_channels, num_features, num_layers_each_block, use_pixel_norm, 
                 use_instance_norm, use_noise):
        super().__init__()
        assert use_instance_norm != use_pixel_norm, 'can only use instance norm or pixel norm'
        self.num_res = num_res

        # convolution for different resolution
        self.conv = []
        for n in range(num_res):
            self.conv.append(nn.ConvTranspose3d(num_channels, num_features[n], kernel_size=3, padding=1))
        self.conv = nn.ModuleList(self.conv)

        # First block
        self.conv1 = nn.Conv3d(num_features[0], num_features[0], kernel_size=3, padding=1)
        self.adaIn1 = AdaIN(num_features[0], use_noise, use_pixel_norm, use_instance_norm)
        self.conv2 = nn.Conv3d(num_features[0], num_features[0], kernel_size=3, padding=1)
        self.adaIn2 = AdaIN(num_features[0], use_noise, use_pixel_norm, use_instance_norm) 
       
        # generate noise, only use for training
        #img_size = 128
        sizes = []
        sizes.append(img_size)
        for _ in range(num_res-1):
            sizes.insert(0, int(sizes[0]/2))
        self.noise = []
        temp = []
        # the first block has 2 layers 
        temp.append(torch.randn(1, 1, sizes[0], sizes[0], sizes[0]).to("cuda"))
        temp.append(torch.randn(1, 1, sizes[0], sizes[0], sizes[0]).to("cuda"))
        #temp.append(torch.zeros(1, 1, sizes[0], sizes[0], sizes[0]).to("cuda"))
        #temp.append(torch.zeros(1, 1, sizes[0], sizes[0], sizes[0]).to("cuda"))
        self.noise.append(temp)
        for r in range(num_res-1):
            temp = []
            for n in range(num_layers_each_block[r+1]):
                temp.append(torch.randn(1, 1, sizes[r+1], sizes[r+1], sizes[r+1]).to("cuda"))
                #temp.append(torch.zeros(1, 1, sizes[r + 1], sizes[r + 1], sizes[r + 1]).to("cuda"))
            self.noise.append(temp)

        # to original number of channels
        self.channel_shrinkage = nn.Conv3d(num_features[-1], int(num_features[-1]/2), kernel_size=3, padding=1)
        self.torgb = nn.Conv3d(int(num_features[-1]/2), num_channels, kernel_size=3, padding=1)

        # generator blocks
        self.GBlocks = []
        for n in range(num_res-1):
            self.GBlocks.append(
                GBlock(num_features[n], num_features[n+1], self.noise[n+1], use_noise, use_pixel_norm,
                       use_instance_norm, num_layers=num_layers_each_block[n+1])
            )
        self.GBlocks = nn.ModuleList(self.GBlocks)

    def forward(self, resolutions):
        for n in range(self.num_res):
            resolutions[n] = self.conv[n](resolutions[n])

        x = resolutions[0]
        x = self.conv1(x)
        x = self.adaIn1(x, self.noise[0][0], resolutions[0])
        x = self.conv2(x)
        x = self.adaIn2(x, self.noise[0][1], resolutions[0])
        for n in range(self.num_res - 1):
            x = self.GBlocks[n](x, resolutions[n+1])
        x = self.channel_shrinkage(x)
        x = self.torgb(x)
        return x
        

class G_mapping(nn.Module):
    def __init__(self,num_res):
        super().__init__()
        """
        Args:
            num_channels(int): the number of channel for filtered data
            num_res(int): how many resolution sizes
            num_features(list): the number of features for each resolution
        """
        self.num_res = num_res
        self.downsample = nn.AvgPool3d(kernel_size=4, stride=2, padding=1)

        """
        self.conv = []
        for n in range(num_res):
            self.conv.append(nn.ConvTranspose3d(num_channels, num_features[n], kernel_size=3, padding=1))
        self.conv = nn.ModuleList(self.conv)
        """

    def forward(self, x):
        """
        Input:
            x(tensor): filtered data
        Return:
            resolutions(list of tensors): list of different resolution datas
        """
        resolutions = []
        resolutions.append(x)
        for _ in range(self.num_res-1):
            resolutions.insert(0, self.downsample(resolutions[0]))

        """
        for n in range(self.num_res):
            resolutions[n] = self.conv[n](resolutions[n])
        """

        return resolutions


class GBlock(nn.Module):
    def __init__(self, previous_num_feature, current_num_feature, noise_input, use_noise, use_pixel_norm, 
                 use_instance_norm, num_layers=2):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(previous_num_feature, current_num_feature, 4, stride=2, padding=1)
        self.num_layers = num_layers
        self.noise_input = noise_input

        self.adaIn = []
        self.conv = []
        for _ in range(num_layers):
            self.adaIn.append(AdaIN(current_num_feature, use_noise, use_pixel_norm, use_instance_norm))
            self.conv.append(nn.Conv3d(current_num_feature, current_num_feature, kernel_size=3, padding=1))
        self.adaIn = nn.ModuleList(self.adaIn)
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x, latent):
        x = self.upsample(x)
        for n in range(self.num_layers):
            x = self.conv[n](x)
            x = self.adaIn[n](x, self.noise_input[n], latent)
        return x

@register_generator_architecture('stylegan_generator_noflatten', 'stylegan_generator_noflatten')
def base_architecture(args):
    args.num_res = getattr(args, 'num_res', 3)
    args.num_features = getattr(args, 'num_features', '256:128:64')
    args.num_layers_each_block = getattr(args, 'num_layers_each_block', '2:2:2')
    args.use_noise = getattr(args, 'use_noise', True)
    args.use_pixel_norm = getattr(args, 'use_pixel_norm', False)
    args.use_instance_norm = getattr(args, 'use_instance_norm', True)


@register_generator_architecture('stylegan_generator_noflatten', 'stylegan_generator_noflatten_progressive')
def base_architecture(args):
    args.num_res = getattr(args, 'num_res', 3)
    args.num_features = getattr(args, 'num_features', '256:128:64')
    args.num_layers_each_block = getattr(args, 'num_layers_each_block', '1:2:3')
    args.use_noise = getattr(args, 'use_noise', True)
    args.use_pixel_norm = getattr(args, 'use_pixel_norm', False)
    args.use_instance_norm = getattr(args, 'use_instance_norm', True)

@register_generator_architecture('stylegan_generator_noflatten', 'stylegan_generator_noflatten_more_resolutions')
def base_architecture(args):
    args.num_res = getattr(args, 'num_res', 4)
    args.num_features = getattr(args, 'num_features', '256:256:128:128')
    args.num_layers_each_block = getattr(args, 'num_layers_each_block', '2:2:2:2')
    args.use_noise = getattr(args, 'use_noise', True)
    args.use_pixel_norm = getattr(args, 'use_pixel_norm', False)
    args.use_instance_norm = getattr(args, 'use_instance_norm', True)

@register_generator_architecture('stylegan_generator_noflatten', 'stylegan_generator_noflatten_more_resolutions_progressive')
def base_architecture(args):
    args.num_res = getattr(args, 'num_res', 4)
    args.num_features = getattr(args, 'num_features', '256:256:128:128')
    args.num_layers_each_block = getattr(args, 'num_layers_each_block', '2:2:4:4')
    args.use_noise = getattr(args, 'use_noise', True)
    args.use_pixel_norm = getattr(args, 'use_pixel_norm', False)
    args.use_instance_norm = getattr(args, 'use_instance_norm', True)
