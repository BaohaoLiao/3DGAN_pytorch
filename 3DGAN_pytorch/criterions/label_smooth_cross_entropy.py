import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class LabelSmoothCrossEntropy(_Loss):
    def __init__(self, label_smoothing=0.1):
        super(LabelSmoothCrossEntropy, self).__init__()
        self.eps = label_smoothing

    def forward(self, logits, kernel, reduce=True):
        lprobs = F.log_softmax(logits, dim=-1)
        kernel = kernel.view(-1, 1)
        nll_loss = -lprobs.gather(dim=-1, index=kernel)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss