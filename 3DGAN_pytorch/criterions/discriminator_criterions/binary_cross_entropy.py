import math
import torch.nn as nn

from . import DiscriminatorCriterion, register_discriminator_criterion


@register_discriminator_criterion('binary_cross_entropy')
class BinaryCrossEntropyCriterion(DiscriminatorCriterion):

    def __init__(self, args):
        super().__init__(args)

    def forward(self, discriminator, real_imgs, gen_imgs, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        d_loss = self.compute_loss(discriminator, real_imgs, gen_imgs, reduce=reduce)
        sample_size = real_imgs.size(0)
        logging_output = {
            'd_loss': d_loss,
            'd_sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, discriminator, real_imgs, gen_imgs, reduce=True):
        assert real_imgs.shape == gen_imgs.shape, 'Unmatched generated image size and real image size'
        valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        BCEloss = nn.BCEloss(reduction='sum')
        real_loss = nn.BCEloss(discriminator(real_imgs), valid)
        fake_loss = nn.BCEloss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2            
        return d_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('d_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('d_sample_size', 0) for log in logging_outputs)
        agg_output = {
            'd_loss': loss_sum / sample_size ,
            'd_sample_size': sample_size,
        }
        return agg_output
