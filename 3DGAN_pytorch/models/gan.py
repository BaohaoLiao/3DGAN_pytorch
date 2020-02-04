import pytorch_lightning as pl

class GAN(pl.LightningModule):
    """Base class for GAN."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add gan-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args):
        """Build a new generator instance."""
        raise NotImplementedError('GANGenerators must implement the build_model method')

    """
    def forward(self, input):
        Args:
            input(LongTensor):
        raise NotImplementedError
    """

