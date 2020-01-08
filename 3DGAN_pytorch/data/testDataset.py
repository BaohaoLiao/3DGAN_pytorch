import glob
import h5py
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class LR2HRTestDataset(Dataset):
    def __init__(self, args):
        self.to_tensor = transforms.ToTensor()
        self.directory = args.test_subset
        self.lr, self.hr = self.get_image()

    def get_image(self):
        files = glob.glob(self.directory+'/*.h5')
        lr = []
        hr = []
        f = files[0]
        data = h5py.File(f, 'r')
        lr_image = np.expand_dims(data['/lr'][:128,:128,:128], axis=0)
        hr_image = np.expand_dims(data['/hr'][:128,:128,:128], axis=0)
        lr.append(self.scale(lr_image))
        hr.append(self.scale(hr_image))
        return lr, hr

    def scale(self, image):
        maximum = np.max(image)
        minimum = np.min(image)
        interval = maximum - minimum
        image = 2.0 * (image - minimum) / (1.0 * interval) - 1.0
        return image


    def __len__(self):
        assert len(self.lr) == len(self.hr), 'unmatched lr and hr length'
        return len(self.lr)

    def __getitem__(self, index):
        lr = torch.from_numpy(self.lr[index]).type(torch.FloatTensor)
        hr = torch.from_numpy(self.hr[index]).type(torch.FloatTensor)
        return lr, hr
        
            
        
    
    
