import importlib
import os

from .generator_criterion import GeneratorCriterion


GENERATOR_CRITERION_REGISTRY = {}
GENERATOR_CRITERION_CLASS_NAMES = set()


def build_criterion(args):
    return GENERATOR_CRITERION_REGISTRY[args.G_criterion].build_criterion(args)


def register_generator_criterion(name):
    """Decorator to register a new generator criterion."""

    def register_criterion_cls(cls):
        if name in GENERATOR_CRITERION_REGISTRY:
            raise ValueError('Cannot register duplicate generator criterion ({})'.format(name))
        if not issubclass(cls, GeneratorCriterion):
            raise ValueError('Criterion ({}: {}) must extend GeneratorCriterion'.format(name, cls.__name__))
        if cls.__name__ in GENERATOR_CRITERION_CLASS_NAMES:
            # We use the criterion class name as a unique identifier in
            # checkpoints, so all criterions must have unique class names.
            raise ValueError('Cannot register generator criterion with duplicate class name ({})'.format(cls.__name__))
        GENERATOR_CRITERION_REGISTRY[name] = cls
        GENERATOR_CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('criterions.generator_criterions.' + module)

