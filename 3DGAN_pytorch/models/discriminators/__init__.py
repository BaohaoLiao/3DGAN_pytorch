import os
import importlib
import argparse

from .gan_discriminator import GANDiscriminator


DISCRIMINATOR_MODEL_REGISTRY = {}
DISCRIMINATOR_ARCH_MODEL_REGISTRY = {}
DISCRIMINATOR_ARCH_MODEL_INV_REGISTRY = {}
DISCRIMINATOR_ARCH_CONFIG_REGISTRY = {}


def build_discriminator(args):
    return DISCRIMINATOR_ARCH_MODEL_REGISTRY[args.D_arch].build_model(args)


def register_discriminator(name):
    """
    New discriminator types can be added with the :func:`register_discriminator` function 
    decorator.

    For example::

        @register_model('DCGANDiscriminator')
        class DCGANGenerator(GANDiscriminator):
            (...)

    .. note:: All models must implement the :class:`GANDiscriminator` interface.

    Args:
        name (str): the name of the discriminator
    """

    def register_discriminator_cls(cls):
        if name in DISCRIMINATOR_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, GANDiscriminator):
            raise ValueError('Model ({}: {}) must extend GANDiscriminator'.format(name, cls.__name__))
        DISCRIMINATOR_MODEL_REGISTRY[name] = cls
        return cls

    return register_discriminator_cls



def register_discriminator_architecture(discriminator_name, arch_name):
    """
    New discriminator architectures can be added with the
    :func:`register_discriminator_architecture` function decorator. After registration,
    discriminator architectures can be selected with the ``--D_arch`` command-line
    argument.

    For example::

        @register_discriminator_architecture('DCGANDiscriminator', 'DCGANDiscriminator_small')
        def DCGANDiscriminator_small(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        discriminator_name (str): the name of the discriminator (Discriminator must already be
            registered)
        arch_name (str): the name of the discriminator architecture (``--arch``)
    """

    def register_discriminator_arch_fn(fn):
        if discriminator_name not in DISCRIMINATOR_MODEL_REGISTRY:
            raise ValueError('Cannot register discriminator architecture for unknown discriminator type ({})'.format(discriminator_name))
        if arch_name in DISCRIMINATOR_ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate DISCRIMINATOR architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Discrimina architecture must be callable ({})'.format(arch_name))
        DISCRIMINATOR_ARCH_MODEL_REGISTRY[arch_name] = DISCRIMINATOR_MODEL_REGISTRY[discriminator_name]
        DISCRIMINATOR_ARCH_MODEL_INV_REGISTRY.setdefault(discriminator_name, []).append(arch_name)
        DISCRIMINATOR_ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_discriminator_arch_fn

# automatically import any Python files in the discriminators/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        discriminator_name = file[:file.find('.py')]
        module = importlib.import_module('models.discriminators.' + discriminator_name)

        # extra `model_parser` for sphinx
        if discriminator_name in DISCRIMINATOR_MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--D_arch', choices=DISCRIMINATOR_ARCH_MODEL_INV_REGISTRY[discriminator_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            DISCRIMINATOR_MODEL_REGISTRY[discriminator_name].add_args(group_args)
            globals()[discriminator_name + '_parser'] = parser

