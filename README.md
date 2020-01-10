# 3DGAN_pytorch
GAN  framework for 3D data

### Requirements
- pytorch >= 1.3 
- lightning-pytorch >=0.5, recommend [custom installation](https://github.com/williamFalcon/pytorch-lightning)


### How to implement
- Check folder 'examples'. StyleGAN model can only be used for gpu training. You can change the noise generation on cpu for cpu traning.
- 'python train.py -h' for more options.
