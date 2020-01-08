import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2,3,4), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2,3,4), True) + self.epsilon)
        return x*tmp
        
