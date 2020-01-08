import os
import importlib
import argparse

from .gan_generator import GANGenerator


GENERATOR_MODEL_REGISTRY = {}
GENERATOR_ARCH_MODEL_REGISTRY = {}
GENERATOR_ARCH_MODEL_INV_REGISTRY = {}
GENERATOR_ARCH_CONFIG_REGISTRY = {}


def build_generator(args):
    return GENERATOR_ARCH_MODEL_REGISTRY[args.G_arch].build_model(args)


def register_generator(name):
    """
    New generator types can be added with the :func:`register_generator` function 
    decorator.

    For example::

        @register_model('DCGANGenerator')
        class DCGANGenerator(GANGenerator):
            (...)

    .. note:: All models must implement the :class:`GANGenerator` interface.

    Args:
        name (str): the name of the generator
    """

    def register_generator_cls(cls):
        if name in GENERATOR_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, GANGenerator):
            raise ValueError('Model ({}: {}) must extend GANGenerator'.format(name, cls.__name__))
        GENERATOR_MODEL_REGISTRY[name] = cls
        return cls

    return register_generator_cls


def register_generator_architecture(generator_name, arch_name):
    """
    New generator architectures can be added with the
    :func:`register_generator_architecture` function decorator. After registration,
    generator architectures can be selected with the ``--G_arch`` command-line
    argument.

    For example::

        @register_generator_architecture('DCGANGenerator', 'DCGANGenerator_small')
        def DCGANGenerator_small(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        generator_name (str): the name of the generator (Generator must already be
            registered)
        arch_name (str): the name of the generator architecture (``--arch``)
    """

    def register_generator_arch_fn(fn):
        if generator_name not in GENERATOR_MODEL_REGISTRY:
            raise ValueError('Cannot register generator architecture for unknown generator type ({})'.format(generator_name))
        if arch_name in GENERATOR_ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate GENERATOR architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Generator architecture must be callable ({})'.format(arch_name))
        GENERATOR_ARCH_MODEL_REGISTRY[arch_name] = GENERATOR_MODEL_REGISTRY[generator_name]
        GENERATOR_ARCH_MODEL_INV_REGISTRY.setdefault(generator_name, []).append(arch_name)
        GENERATOR_ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn
    return register_generator_arch_fn


# automatically import any Python files in the generators/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        generator_name = file[:file.find('.py')]
        module = importlib.import_module('models.generators.' + generator_name)

        # extra `model_parser` for sphinx
        if generator_name in GENERATOR_MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--G_arch', choices=GENERATOR_ARCH_MODEL_INV_REGISTRY[generator_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            GENERATOR_MODEL_REGISTRY[generator_name].add_args(group_args)
            globals()[generator_name + '_parser'] = parser

