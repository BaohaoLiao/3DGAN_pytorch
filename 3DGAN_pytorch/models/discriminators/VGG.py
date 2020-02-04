import torch.nn as nn

from . import (
    GANDiscriminator, register_discriminator, register_discriminator_architecture,
)


@register_discriminator('vgg')
class VGG(GANDiscriminator):
    def __init__(self, args):
        super().__init__()
        self.features = Features(args.num_channels)
        self.dense_layer = FC()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        return VGG(args)

    def forward(self, x):
        out = self.features(x)
        out = self.dense_layer(out)
        return out

class Features(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),   # 32 -> 16
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 16 -> 8
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 8 -> 4
        )

    def forward(self, x):
        out = self.features(x)
        return out

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(4*4*4*256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.compress(out)
        return out


@register_discriminator_architecture('vgg', 'vgg')
def base_architecture(args):
    return