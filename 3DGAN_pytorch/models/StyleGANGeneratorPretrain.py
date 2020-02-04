import os
from collections import OrderedDict
import h5py
import matplotlib.pyplot as plt

import torch
import torchvision
import pytorch_lightning as pl

from models import generators
from data.trainDataset import LR2HRTrainDataset
from data.validDataset import LR2HRValidDataset
from data.testDataset import LR2HRTestDataset
from criterions.pixel_loss import pixel_loss
from . import (
    GAN, register_gan, register_gan_architecture,
)

@register_gan('stylegan_generator_pretrain')
class StyleGANGeneratorPretrain(GAN):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        print(self.hparams)

        #build model
        self.generator = generators.build_generator(args)

        if self.hparams.train_or_test != 'test':
            print(self.generator)
            print('| GAN: {}'.format(args.model))
            print('| generator: {}'.format(args.G_arch))
            print('| num. generator params: {} (num. trained: {})'.format(
                sum(p.numel() for p in self.generator.parameters()),
                sum(p.numel() for p in self.generator.parameters() if p.requires_grad),
            ))

        self.pixel_loss = pixel_loss(loss_type=self.hparams.pixel_loss_type)
        self.val_loss = pixel_loss(loss_type='l1')
 
        # cache for generated images
        self.generated_imgs = None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--pixel-loss-type', type=str, default='l1')
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new gan instance."""
        base_architecture(args) 
        return StyleGANGeneratorPretrain(args)

    def forward(self, z):
        return self.generator(z)
    
    def training_step(self, batch, batch_idx):
        #print(self.epoch, self.global_step, self.early_stop_callback_patience)
        lr, hr = batch
        fake_img = self.forward(lr)

        pixel_loss = self.pixel_loss(fake_img, hr)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            pixel_loss = pixel_loss.unsqueeze(0)

        tqdm_dict = {'pixel_loss': pixel_loss}
        output = OrderedDict({
            'loss': pixel_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        self.generated_imgs = self.forward(lr)
        val_loss = self.val_loss(self.generated_imgs, hr)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            val_loss = val_loss.unsqueeze(0)

        output = OrderedDict({
            'val_loss': val_loss,
        })       
        
        return output
    
    def validation_end(self, outputs):
        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}

        grid = torchvision.utils.make_grid(self.generated_imgs[0][0 ,0, :, :])
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        return result

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        fake_img = self.forward(lr)
        test_loss = self.val_loss(fake_img, hr)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            test_loss = test_loss.unsqueeze(0)

        output = OrderedDict({
            'test_loss': test_loss,
        })
  
        # save test images
        img_blocks = [lr[0][0, :, :, :].to('cpu'), hr[0][0, :, :, :].to('cpu'), fake_img[0][0, :, :, :].to('cpu')]
        self.save_images(img_blocks)
        imgs = [lr[0][0, 0, :, :].to('cpu'), hr[0][0, 0, :, :].to('cpu'), fake_img[0][0, 0, :, :].to('cpu')]
        self.image_show(imgs)
        return output

    def test_end(self, outputs):
        test_loss_mean = 0
        for output in outputs:
            test_loss = output['test_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

        test_loss_mean /= len(outputs)
        tqdm_dict = {'test_loss': test_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
        return result

    def configure_optimizers(self):
        G_opt = torch.optim.Adam(self.generator.parameters(), lr=float(self.hparams.G_lr))
        G_sched = torch.optim.lr_scheduler.ExponentialLR(G_opt, gamma=0.99)
        return [G_opt], [G_sched]

    @pl.data_loader
    def train_dataloader(self):
        dataset = LR2HRTrainDataset(self.hparams)
        train_sampler = None
        if self.use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        #should_shuffle = train_sampler is None

        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, 
                             batch_size=self.hparams.batch_size, shuffle=True,
                             num_workers=self.hparams.num_workers)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataset = LR2HRValidDataset(self.hparams)
        valid_sampler = None
        if self.use_ddp:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, 
                             batch_size=self.hparams.batch_size, shuffle=False,
                             num_workers=self.hparams.num_workers)
        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        dataset = LR2HRTestDataset(self.hparams)
        test_sampler = None
        if self.use_ddp:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler,
                             batch_size=self.hparams.batch_size, shuffle=False,
                             num_workers=self.hparams.num_workers)
        return dataloader

    def save_images(self, images):
        f = os.path.join(self.hparams.save_test_images, 'generated.h5')
        hf = h5py.File(f, 'w')
        hf.create_dataset('/lr', data=images[0])
        hf.create_dataset('/hr', data=images[1])
        hf.create_dataset('/gen', data=images[2])
        hf.close

    def image_show(self, imgs):
        """
        imgs: lr, hr, fake
        """
        fig = plt.figure()
        a = fig.add_subplot(1,3,1)
        plt.imshow(imgs[0])
        a.set_title('input')
        plt.axis('off')
        b = fig.add_subplot(1,3,2)
        plt.imshow(imgs[1])
        b.set_title('ground truth')
        plt.axis('off')
        c = fig.add_subplot(1,3,3)
        plt.imshow(imgs[2])
        c.set_title('generated')
        plt.axis('off')
        
        f = os.path.join(self.hparams.save_test_images, 'generated.png')
        plt.savefig(f)
 
@register_gan_architecture('stylegan_generator_pretrain', 'stylegan_generator_pretrain')
def base_architecture(args):
    args.pixel_loss_type = getattr(args, 'pixel_loss_type', 'l1')

    
