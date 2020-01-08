import torch.nn as nn

class GANDiscriminator(nn.Module):
    """Base class for discriminators."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add discriminator-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new discriminator instance."""
        raise NotImplementedError('GANDiscriminators must implement the build_model method')

    def forward(self, x):
        """
        Args:
            x (Tensor): output from generator or targets.
        """
        raise NotImplementedError

