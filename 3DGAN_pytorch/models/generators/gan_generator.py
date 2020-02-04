import torch.nn as nn

class GANGenerator(nn.Module):
    """Base class for generators."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add generator-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""
        raise NotImplementedError('GANGenerators must implement the build_model method')

    """
    def forward(self, input):
        #Args:
        #    input(LongTensor):
        raise NotImplementedError
    """

