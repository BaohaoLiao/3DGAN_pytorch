import torch
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
        super(StyleGANDiscriminator, self).__init__()

        features = args.D_num_features.split(':')
        assert len(features) == args.num_res, 'number of discriminator features does not match number of resolution'
        num_features = list(map(int, features))

        self.features = Features(args.num_channels, num_features)
        self.compress = Compress(num_features)

        self.multitask = args.multitask
        if self.multitask:
            self.classification = Classification(num_features, args.mapping_fmaps)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--D-num-features', type=str, metavar='N')
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        
        return StyleGANDiscriminator(args)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.compress(x)
        if self.multitask:
            logits = self.classification(x)
            return out, logits
        else:
            return out


class Features(nn.Module):
    def __init__(self, num_channels, num_features):
        super(Features, self).__init__()
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv3d(num_channels, num_features[0], kernel_size=1)

        # TODO blur

        # down sample
        self.down1 = nn.Conv3d(num_features[0], num_features[1], kernel_size=2, stride=2)
        self.down2 = nn.Conv3d(num_features[1], num_features[2], kernel_size=2, stride=2)
        self.down3 = nn.Conv3d(num_features[2], num_features[3], kernel_size=2, stride=2)

        # conv: padding=same
        self.conv11 = nn.Conv3d(num_features[0], num_features[0], kernel_size=3, padding=1)
        self.conv12 = nn.Conv3d(num_features[0], num_features[0], kernel_size=3, padding=1)
        self.conv13 = nn.Conv3d(num_features[0], num_features[0], kernel_size=3, padding=1)
        self.conv21 = nn.Conv3d(num_features[1], num_features[1], kernel_size=3, padding=1)
        self.conv22 = nn.Conv3d(num_features[1], num_features[1], kernel_size=3, padding=1)
        self.conv23 = nn.Conv3d(num_features[1], num_features[1], kernel_size=3, padding=1)
        self.conv31 = nn.Conv3d(num_features[2], num_features[2], kernel_size=3, padding=1)
        self.conv32 = nn.Conv3d(num_features[2], num_features[2], kernel_size=3, padding=1)
        self.conv33 = nn.Conv3d(num_features[2], num_features[2], kernel_size=3, padding=1)
        self.conv_last = nn.Conv3d(num_features[3], num_features[3], kernel_size=3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fromrgb(x), 0.2, inplace=True)
        # 1. 32 x 32 x 32 -> 16 x 16 x 16
        x = F.leaky_relu(self.conv11(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv12(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv13(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down1(x), 0.2, inplace=True)
        #2. 16 x 16 x 16 -> 8 x 8 x 8
        x = F.leaky_relu(self.conv21(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv22(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv23(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down2(x), 0.2, inplace=True)
        #3. 8 x 8 x 8 -> 4 x 4 x 4
        x = F.leaky_relu(self.conv31(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv32(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv33(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
        return x

class Compress(nn.Module):
    def __init__(self, num_features):
        super(Compress, self).__init__()
        self.compress = nn.Sequential(
            nn.Linear(num_features[3] * (4 ** 3), num_features[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_features[3], num_features[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_features[3], num_features[3] // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_features[3] // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.compress(x)
        return out

class Classification(nn.Module):
    def __init__(self, num_features, mapping_fmaps):
        super(Classification, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(num_features[3] * (4 ** 3), num_features[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_features[3], num_features[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_features[3], mapping_fmaps),
        )
        self.embed_out = nn.Parameter(torch.Tensor(8, mapping_fmaps))
        nn.init.normal_(self.embed_out, mean=0, std=mapping_fmaps ** -0.5)

    def forward(self, x):
        out = self.classification(x)
        out = F.linear(out, self.embed_out)
        return out

@register_discriminator_architecture('stylegan_discriminator', 'stylegan_discriminator')
def base_architecture(args):
    return


