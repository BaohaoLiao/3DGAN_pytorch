import os
import importlib
import argparse

from .gan import GAN


GAN_MODEL_REGISTRY = {}
GAN_ARCH_MODEL_REGISTRY = {}
GAN_ARCH_MODEL_INV_REGISTRY = {}
GAN_ARCH_CONFIG_REGISTRY = {}


def build_gan(args):
    return GAN_ARCH_MODEL_REGISTRY[args.model].build_model(args)


def register_gan(name):
    def register_gan_cls(cls):
        if name in GAN_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, GAN):
            raise ValueError('Model ({}: {}) must extend GAN'.format(name, cls.__name__))
        GAN_MODEL_REGISTRY[name] = cls
        return cls

    return register_gan_cls


def register_gan_architecture(gan_name, arch_name):
    def register_gan_arch_fn(fn):
        if gan_name not in GAN_MODEL_REGISTRY:
            raise ValueError('Cannot register gan architecture for unknown gan type ({})'.format(gan_name))
        if arch_name in GAN_ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate gan architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('GAN architecture must be callable ({})'.format(arch_name))
        GAN_ARCH_MODEL_REGISTRY[arch_name] = GAN_MODEL_REGISTRY[gan_name]
        GAN_ARCH_MODEL_INV_REGISTRY.setdefault(gan_name, []).append(arch_name)
        GAN_ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn
    return register_gan_arch_fn


# automatically import any Python files in the generators/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        gan_name = file[:file.find('.py')]
        module = importlib.import_module('models.' + gan_name)

        # extra `model_parser` for sphinx
        if gan_name in GAN_MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--model', choices=GAN_ARCH_MODEL_INV_REGISTRY[gan_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            GAN_MODEL_REGISTRY[gan_name].add_args(group_args)
            globals()[gan_name + '_parser'] = parser

