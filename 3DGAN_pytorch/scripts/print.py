import torch
import re

checkpoint = torch.load('_ckpt_epoch_best.ckpt', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']

for key in list(checkpoint.keys()):
    print(key)
