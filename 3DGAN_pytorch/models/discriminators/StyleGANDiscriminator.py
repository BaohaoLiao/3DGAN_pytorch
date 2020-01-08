import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    GANDiscriminator, register_discriminator, register_discriminator_architecture,
)


@register_discriminator('stylegan_discriminator')
class StyleGANDiscriminator(GANDiscriminator):
    """
    DCGAN discriminator consisting of *args.discriminator_layers* layers. Each layer
    is a :class:`DCGANDiscriminatorLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        input: a batch of generator output or high resolution images
    """

    def __init__(self, args):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--fmaps-base', type=int, default=4096, metavar='M')
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        
        return Discriminator(args)


class Discriminator(nn.Module):
    def __init__(self, args, structure='fixed', fmap_max=256, fmap_decay=1.0, f=None):
        super().__init__()
        fmap_base = args.fmap_base
        self.resolution = args.img_size
        self.resolution_log2 = int(np.log2(args.img_size))
        assert self.resolution == 2 ** self.resolution_log2 and self.resolution >=4
        #self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv3d(args.channels, self.nf(self.resolution_log2-1), kernel_size=1)
        self.structure = structure
      
        #TODO blur

        # down sample
        self.down1 = nn.Conv3d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), 
                                kernel_size=2, stride=2)
        self.down2 = nn.Conv3d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-2),  
                                kernel_size=2, stride=2)
        self.down3 = nn.Conv3d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-3),
                                kernel_size=2, stride=2)

        # conv: padding=same
        self.conv1 = nn.Conv3d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), 
                               kernel_size=3, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), 
                               kernel_size=3, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), 
                               kernel_size=3, padding=(1, 1, 1))

        # calculate point:
        self.conv_last = nn.Conv3d(self.nf(self.resolution_log2-3), self.nf(1), kernel_size=3, 
                                   padding=(1, 1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def nf(self, idx):
        nf = min(int(self.fmap_base / (2.0 ** (idx * self.fmap_decay))), self.fmap_max)
        return nf

    def forward(self, x):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(x), 0.2, inplace=True)
            # 1. 32 x 32 x 32 x nf(4) -> 16 x 16 x 16
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(x), 0.2, inplace=True)
            #2. 16 x 16 x 16 -> 8 x 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down2(x), 0.2, inplace=True)
            #3. 8 x 8 x 8 -> 4 x 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down3(x), 0.2, inplace=True)
            #4. 4 x 4 x 4 -> 1
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 4096(4 x 4 x 4 x nf(1))
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x
 

@register_discriminator_architecture('stylegan_discriminator', 'stylegan_discriminator_32')
def base_architecture(args):
    args.fmap_base = getattr(args, 'fmap_base', 16384)

@register_discriminator_architecture('stylegan_discriminator', 'stylegan_discriminator_16')
def base_architecture(args):
    args.fmap_base = getattr(args, 'fmap_base', 2048)
