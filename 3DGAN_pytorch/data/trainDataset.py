import glob
import h5py
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class LR2HRTrainDataset(Dataset):
    def __init__(self, args):
        self.to_tensor = transforms.ToTensor()
        self.directory = args.train_subset
        self.img_size = args.img_size
        self.lr, self.hr, self.kernel = self.crop()

    def crop(self):
        datas = self.directory.split(':')
        lr = []
        hr = []
        kernel = []
        for data in datas:
            files = glob.glob(data+'/*.h5')
            for f in files:
                data = h5py.File(f, 'r')
                shape =  np.shape(data['/lr'])
                for i in range(shape[0] // (2 * self.img_size)):
                    for j in range(shape[1] //  (2 * self.img_size)):
                        for k in range(shape[2] //  (2 * self.img_size)):
                            lr_image = np.expand_dims(data['/lr'][i*2*self.img_size:(i+1)*2*self.img_size,
                                                  j*2*self.img_size:(j+1)*2*self.img_size,
                                                  k*2*self.img_size:(k+1)*2*self.img_size], axis=0)
                            hr_image = np.expand_dims(data['/hr'][i*2*self.img_size:(i+1)*2*self.img_size,
                                                  j*2*self.img_size:(j+1)*2*self.img_size,
                                                  k*2*self.img_size:(k+1)*2*self.img_size], axis=0)

                            kernel += data['/kernel']
                            lr.append(lr_image)
                            hr.append(hr_image)
        return lr, hr, kernel

    def scale(self, image):
        maximum = np.max(image)
        minimum = np.min(image)
        interval = maximum - minimum
        image = 2.0 * (image - minimum) / (1.0 * interval) - 1.0
        return image

    def __len__(self):
        assert len(self.lr) == len(self.hr) == len(self.kernel), 'unmatched lr, hr and kernel length'
        return len(self.lr)

    def randomCrop(self, image):
        top = np.random.randint(0,  self.img_size)
        left = np.random.randint(0, self.img_size)
        front = np.random.randint(0, self.img_size)
        out = image[:, top:top+self.img_size, left:left+self.img_size, front:front+self.img_size]
        return out

    def __getitem__(self, index):
        lr = self.lr[index]
        hr = self.hr[index]

        # RandomCrop
        lr = self.randomCrop(lr)
        hr = self.randomCrop(hr)

        # Rescale
        lr = self.scale(lr)
        hr = self.scale(hr)

        # to Tensor
        lr = torch.from_numpy(lr).type(torch.FloatTensor)
        hr = torch.from_numpy(hr).type(torch.FloatTensor)
        #kernel = torch.from_numpy(self.kernel[index]).type(torch.IntTensor)
        kernel = torch.as_tensor(self.kernel[index])
        return lr, hr, kernel
        
            
        
    
    
