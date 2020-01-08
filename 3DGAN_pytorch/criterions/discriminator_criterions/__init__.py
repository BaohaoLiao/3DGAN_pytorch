import importlib
import os

from .discriminator_criterion import DiscriminatorCriterion


DISCRIMINATOR_CRITERION_REGISTRY = {}
DISCRIMINATOR_CRITERION_CLASS_NAMES = set()


def build_criterion(args):
    return DISCRIMINATOR_CRITERION_REGISTRY[args.D_criterion].build_criterion(args)


def register_discriminator_criterion(name):
    """Decorator to register a new discriminator criterion."""

    def register_criterion_cls(cls):
        if name in DISCRIMINATOR_CRITERION_REGISTRY:
            raise ValueError('Cannot register duplicate discriminator criterion ({})'.format(name))
        if not issubclass(cls, DiscriminatorCriterion):
            raise ValueError('Criterion ({}: {}) must extend DiscriminatorCriterion'.format(name, cls.__name__))
        if cls.__name__ in DISCRIMINATOR_CRITERION_CLASS_NAMES:
            # We use the criterion class name as a unique identifier in
            # checkpoints, so all criterions must have unique class names.
            raise ValueError('Cannot register discriminator criterion with duplicate class name ({})'.format(cls.__name__))
        DISCRIMINATOR_CRITERION_REGISTRY[name] = cls
        DISCRIMINATOR_CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/discriminator_criterions directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('criterions.discriminator_criterions.' + module)

