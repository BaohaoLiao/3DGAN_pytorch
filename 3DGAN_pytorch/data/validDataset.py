import glob
import h5py
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class LR2HRValidDataset(Dataset):
    def __init__(self, args):
        self.to_tensor = transforms.ToTensor()
        self.directory = args.train_subset
        self.img_size = args.img_size
        self.lr, self.hr = self.crop()

    def crop(self):
        datas = self.directory.split(':')
        lr = []
        hr = []
        for data in datas:
            files = glob.glob(data + '/*.h5')
            for f in files:
                data = h5py.File(f, 'r')
                shape =  np.shape(data['/lr'])
                i = shape[0] // self.img_size - 1
                j = shape[0] // self.img_size - 1
                k = shape[0] // self.img_size - 1
                lr_image = np.expand_dims(data['/lr'][i*self.img_size:(i+1)*self.img_size,
                                      j*self.img_size:(j+1)*self.img_size,
                                      k*self.img_size:(k+1)*self.img_size], axis=0)
                hr_image = np.expand_dims(data['/hr'][i*self.img_size:(i+1)*self.img_size,
                                      j*self.img_size:(j+1)*self.img_size,
                                      k*self.img_size:(k+1)*self.img_size], axis=0)
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
        
            
        
    
    
