import math
import torch.nn as nn

from . import GeneratorCriterion, register_generator_criterion


@register_generator_criterion('binary_cross_entropy')
class BinaryCrossEntropyCriterion(GeneratorCriterion):

    def __init__(self, args):
        super().__init__(args)

    def forward(self, discriminator, gen_imgs, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = self.compute_loss(discriminator, gen_imgs, reduce=reduce)
        sample_size = gen_imgs.size(0)
        logging_output = {
            'g_loss': loss,
            'g_sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, discriminator, gen_imgs, reduce=True):
        valid = Variable(Tensor(gen_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        BCEloss = nn.BCEloss(reduction='sum')
        g_loss = BCEloss(discriminator(gen_imgs), valid)
        return g_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('g_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('g_sample_size', 0) for log in logging_outputs)
        agg_output = {
            'g_loss': loss_sum / sample_size ,
            'g_sample_size': sample_size,
        }
        return agg_output
