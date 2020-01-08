from torch.nn.modules.loss import _Loss


class GeneratorCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, args):
        return cls(args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)

