import torch
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

class adversarial_loss(_Loss):
    def __init__(self, discriminator):
        super().__init__()
        self.D_net = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, img, mode='opt_G', real_loss=True):
        if mode == 'opt_G':
            valid = torch.ones(img.size(0), 1).to(img.device)
            loss = self.loss(self.D_net(img), valid)
        if mode == 'opt_D':
            if real_loss:
                valid = torch.ones(img.size(0), 1).to(img.device)
                loss = self.loss(self.D_net(img), valid)
            else:
                fake = torch.zeros(img.size(0), 1).to(img.device)
                loss = self.loss(self.D_net(img.detach()), fake)
        return loss




            


            
            
            
        
    
