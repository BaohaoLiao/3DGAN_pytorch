import os
from collections import OrderedDict
import h5py
import matplotlib.pyplot as plt

import torch
import torchvision
import pytorch_lightning as pl

from models import generators, discriminators
from data.trainDataset import LR2HRTrainDataset
from data.validDataset import LR2HRValidDataset
from data.testDataset import LR2HRTestDataset
from criterions.pixel_loss import pixel_loss
from criterions.adversarial_loss import adversarial_loss
from criterions.label_smooth_cross_entropy import LabelSmoothCrossEntropy
from . import (
    GAN, register_gan, register_gan_architecture,
)

@register_gan('stylegan')
class StyleGAN(GAN):
    def __init__(self, args):
        super(StyleGAN, self).__init__()
        self.hparams = args
        print(self.hparams)

        #build model
        self.generator = generators.build_generator(args)
        self.discriminator = discriminators.build_discriminator(args)
        print(self.generator)
        print(self.discriminator)
        print('| GAN: {}'.format(args.model))
        print('| generator: {}'.format(args.G_arch))
        print('| num. generator params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.generator.parameters()),
            sum(p.numel() for p in self.generator.parameters() if p.requires_grad),
        ))
        print('| num. generator-mapping params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.generator.mapping.parameters()),
            sum(p.numel() for p in self.generator.mapping.parameters() if p.requires_grad),
        ))
        print('| num. generator-projection params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.generator.projection.parameters()),
            sum(p.numel() for p in self.generator.projection.parameters() if p.requires_grad),
        ))
        print('| num. generator-synthesis params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.generator.synthesis.parameters()),
            sum(p.numel() for p in self.generator.synthesis.parameters() if p.requires_grad),
        ))
        print('| discriminator: {}'.format(args.D_arch))
        print('| num. discriminator params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.discriminator.parameters()),
            sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad),
        ))

        if self.hparams.pixel_loss:
            self.pixel_loss = pixel_loss(loss_type=self.hparams.pixel_loss_type)
            self.pixel_loss_weight = self.hparams.pixel_loss_weight
        self.adversarial_loss = adversarial_loss()
        if self.hparams.multitask:
            self.classificationLoss = LabelSmoothCrossEntropy(self.hparams.label_smoothing)
        self.val_loss = pixel_loss(loss_type='l1')
 
        # cache for generated images
        self.generated_imgs = None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--known-filter', action='store_true', default=False)
        parser.add_argument('--mapping-fmaps', type=int, metavar='M')
        parser.add_argument('--pixel-loss', action='store_true', default=False,
                            help='whether to use pixel loss')
        parser.add_argument('--pixel-loss-type', type=str, default='l1',
                            help='what kind of pixel loss (l1 or l2) to use')
        parser.add_argument('--pixel-loss-weight', type=float)
        parser.add_argument('--multitask', action='store_true', default=False,
                            help='whether to add label classification task as a auxiliary task')
        parser.add_argument('--multitask-loss-weight', type=float, default=0.1)
        parser.add_argument('--label-smoothing', type=float, default=0.1)
        parser.add_argument('--percentage', type=float, default=0)
        # fmt: on

    @classmethod
    def build_model(cls, args):
        """Build a new gan instance."""
        base_architecture(args) 
        return StyleGAN(args)

    def forward(self, z, lr):
        return self.generator(z, lr)

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr, hr, kernel = batch
        if self.hparams.known_filter:
            uniform = torch.rand(lr.size(0)).to(lr.device)
            random_select = uniform.ge(1. - self.hparams.percentage)
            replaced_kernel = kernel.masked_fill(random_select, 7)

        if self.hparams.known_filter:
            fake_img = self.forward(replaced_kernel, lr)
        else:
            z = torch.randn([lr.size(0), self.hparams.mapping_fmaps]).to(lr.device)
            fake_img = self.forward(z, lr)
        

        # train discriminator
        if optimizer_idx == 1:
            #real_loss = self.adversarial_loss(hr, mode='opt_D', real_loss=True)
            #fake_loss = self.adversarial_loss(fake_img, mode='opt_D', real_loss=False)
            if self.hparams.multitask:
                true_compress, true_logits = self.discriminator(hr)
                fake_compress, _ = self.discriminator(fake_img.detach())
                true_classification_loss = self.classificationLoss(true_logits, kernel)
            else:
                true_compress = self.discriminator(hr)
                fake_compress = self.discriminator(fake_img.detach())

            real_loss = self.adversarial_loss(true_compress, mode='opt_D', real_loss=True)
            fake_loss = self.adversarial_loss(fake_compress, mode='opt_D', real_loss=False)
            d_loss = (real_loss + fake_loss) / 2.0

            # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
            if self.trainer.use_dp or self.trainer.use_ddp2:
                d_loss = d_loss.unsqueeze(0)
                if self.hparams.multitask:
                    ture_classification_loss = true_classification_loss.unsqueeze(0)
            loss = d_loss

            if self.hparams.multitask:
                loss = loss + true_classification_loss
                tqdm_dict = {'d_loss': d_loss, 'true_class_loss': ture_classification_loss}
            else:
                tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train generator
        if optimizer_idx == 0:
            if self.hparams.multitask:
                fake_compress, fake_logits = self.discriminator(fake_img)
                fake_classification_loss = self.classificationLoss(fake_logits, kernel)
            else:
                fake_compress = self.discriminator(fake_img)

            g_loss = self.adversarial_loss(fake_compress, mode='opt_G', real_loss=True)

            if self.hparams.pixel_loss:
                pixel_loss = self.pixel_loss(fake_img, hr)

            # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
            if self.trainer.use_dp or self.trainer.use_ddp2:
                g_loss = g_loss.unsqueeze(0)
                if self.hparams.pixel_loss:
                    pixel_loss = pixel_loss.unsqueeze(0)
                if self.hparams.multitask:
                    fake_classification_loss = fake_classification_loss.unsqueeze(0)

            loss = g_loss
            if self.hparams.pixel_loss:
                loss += self.hparams.pixel_loss_weight * pixel_loss
            if self.hparams.multitask:
                loss += self.hparams.multitask_loss_weight * fake_classification_loss

            if self.global_step % self.hparams.critic_iter != 0:
                loss = loss * 0.0

            if self.hparams.pixel_loss and self.hparams.multitask:
                tqdm_dict = {'g_loss': g_loss, 'pixel_loss': pixel_loss, 'fake_class_loss': fake_classification_loss}
            elif self.hparams.pixel_loss and not self.hparams.multitask:
                tqdm_dict = {'g_loss': g_loss, 'pixel_loss': pixel_loss}
            elif not self.hparams.pixel_loss and self.hparams.multitask:
                tqdm_dict = {'g_loss': g_loss, 'fake_class_loss': fake_classification_loss}
            else:
                tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        lr, hr, kernel = batch
        if self.hparams.known_filter:
            self.generated_imgs = self.forward(kernel, lr)
        else:
            z = torch.randn([lr.size(0), self.hparams.mapping_fmaps]).to(lr.device)
            self.generated_imgs = self.forward(z, lr)
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
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        return result

    def test_step(self, batch, batch_idx):
        lr, hr, kernel = batch
        if self.hparams.known_filter:
            fake_img = self.forward(kernel, lr)
        else:
            z = torch.randn([lr.size(0), self.hparams.mapping_fmaps]).to(lr.device)
            fake_img = self.forward(z, lr)
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
        D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=float(self.hparams.D_lr))
        G_sched = torch.optim.lr_scheduler.ExponentialLR(G_opt, gamma=0.99)
        D_sched = torch.optim.lr_scheduler.ExponentialLR(D_opt, gamma=0.99)
        return [G_opt, D_opt], [G_sched, D_sched]

    @pl.data_loader
    def train_dataloader(self):
        dataset = LR2HRTrainDataset(self.hparams)
        train_sampler = None
        if self.use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        should_shuffle = train_sampler is None

        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, 
                             batch_size=self.hparams.batch_size, shuffle=should_shuffle,
                             num_workers=self.hparams.num_workers)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataset = LR2HRValidDataset(self.hparams)
        valid_sampler = None
        if self.use_ddp:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
  
        should_shuffle = None
        dataloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, 
                             batch_size=self.hparams.batch_size, shuffle=should_shuffle,
                             num_workers=self.hparams.num_workers)
        return dataloader

    @pl.data_loader
    def test_dataloader(self):
        dataset = LR2HRTestDataset(self.hparams)
        test_sampler = None
        if self.use_ddp:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        should_shuffle = None
        dataloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler,
                             batch_size=self.hparams.batch_size, shuffle=should_shuffle,
                             num_workers=self.hparams.num_workers)
        return dataloader

    def save_images(self, images):
        f = os.path.join(self.hparams.save_test_images, 'generated.h5')
        hf = h5py.File(f, 'w')
        #hf.create_dataset('/lr', data=images[0])
        #hf.create_dataset('/hr', data=images[1])
        hf.create_dataset('/gen', data=images[2])
        hf.close

    def image_show(self, imgs):
        """
        imgs: lr, hr, fake
        """
        fig = plt.figure()
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(imgs[0])
        a.set_title('input')
        plt.axis('off')
        b = fig.add_subplot(1, 3, 2)
        plt.imshow(imgs[1])
        b.set_title('ground truth')
        plt.axis('off')
        c = fig.add_subplot(1, 3, 3)
        plt.imshow(imgs[2])
        c.set_title('generated')
        plt.axis('off')

        f = os.path.join(self.hparams.save_test_images, 'generated.png')
        plt.savefig(f)
 
@register_gan_architecture('stylegan', 'stylegan')
def base_architecture(args):
    return
    
