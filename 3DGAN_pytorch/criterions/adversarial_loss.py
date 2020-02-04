import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class adversarial_loss(_Loss):
    def __init__(self):
        super(adversarial_loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, compress, mode='opt_G', real_loss=True):
        if mode == 'opt_G':
            valid = torch.ones(compress.size(0), 1).to(compress.device)
            loss = self.loss(compress, valid)
        if mode == 'opt_D':
            if real_loss:
                valid = torch.ones(compress.size(0), 1).to(compress.device)
                loss = self.loss(compress, valid)
            else:
                fake = torch.zeros(compress.size(0), 1).to(compress.device)
                loss = self.loss(compress, fake)
        return loss