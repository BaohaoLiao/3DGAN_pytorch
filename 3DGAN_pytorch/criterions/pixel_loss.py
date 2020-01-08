import torch.nn as nn
from torch.nn.modules.loss import _Loss


class pixel_loss(_Loss):
    def __init__(self, loss_type):
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        if loss_type == 'l2':
            self.loss = nn.MSELoss()

    def forward(self, fake_img, real_img):
        x = self.loss(fake_img, real_img)
        return x