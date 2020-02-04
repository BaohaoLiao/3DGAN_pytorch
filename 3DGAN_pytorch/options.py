import argparse
import torch

from models.generators import GENERATOR_ARCH_MODEL_REGISTRY, GENERATOR_ARCH_CONFIG_REGISTRY
from models.discriminators import DISCRIMINATOR_ARCH_MODEL_REGISTRY, DISCRIMINATOR_ARCH_CONFIG_REGISTRY
from models import GAN_ARCH_MODEL_REGISTRY, GAN_ARCH_CONFIG_REGISTRY

def get_all_parser():
    parser = get_parser()
    add_dataset_args(parser, train=True)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser

def get_training_parser():
    parser = get_parser()
    add_dataset_args(parser, train=True)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser

def get_test_parser():
    parser = get_parser()
    add_dataset_args(parser, test=True)
    add_model_args(parser)
    return parser

def get_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument('--train-or-test', type=str)
    parser.add_argument('--progress-bar', action='store_true', default=False,
                        help='whether to disable progress bar')
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--num-gpu', type=int,
                         help='how many gpus to use (one node)')
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, test=False):
    group = parser.add_argument_group('Dataset and data loading')
    # fmt: off
    group.add_argument('--num-workers', default=0, type=int, metavar='N',
                       help='how many subprocesses to use for data loading')
    group.add_argument('--batch-size', type=int, metavar='N',
                       help='maximum image in a batch')
    group.add_argument("--img-size", type=int,
                       help="size of each image dimeansion") #change to patch size
    group.add_argument("--num-channels", type=int, default=1,
                       help="number of image channels")
    #if train:
    group.add_argument('--train-subset', default='train', metavar='SPLIT',
                        help='data subset to use for training (train, valid, test)')
    group.add_argument('--train-subset-split', type=float, default=1.0, metavar='SPLIT',
                        help='split the train subset if too big')
    group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                        help='comma separated list of data subsets to use for validation'
                            ' (train, valid, valid1, test, test1)')
    group.add_argument('--monitor', type=str, metavar='M', required=True,
                        help='monitor to decide which checkpoint to save')
    group.add_argument('--mode', type=str, metavar='M', default='min',
                        choices=['min', 'auto', 'max'],
                        help='decide the loss trend')
    group.add_argument('--patience', type=int, metavar='P',
                        help='when to stop training early')
    group.add_argument('--critic-iter', default=1, type=int, metavar='C',
                        help='how often to train generator')
    #if test:
    group.add_argument('--test-subset', default='test', metavar='SPLIT',
                        help='data subset to generate (train, valid, test)')
    group.add_argument('--save-test-images', type=str,  default='test_images',
                        help='where to save generated data')
    group.add_argument('--restore-checkpoint', type=str, metavar='R',
                        help='use which checkpoint to generate test data')
    group.add_argument('--meta-tags', type=str, metavar='T',
                        help='tag information in version')
    # fmt: on
    return group


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')
    # fmt: off
    group.add_argument('--model', metavar='MODEL', required=True,
                       choices=GAN_ARCH_MODEL_REGISTRY.keys(),
                       help='GAN Architecture')
    group.add_argument('--G_arch', metavar='ARCH', required=True,
                       choices=GENERATOR_ARCH_MODEL_REGISTRY.keys(),
                       help='Generator Architecture')
    group.add_argument('--D_arch', metavar='ARCH', required=True,
                       choices=DISCRIMINATOR_ARCH_MODEL_REGISTRY.keys(),
                       help='Discriminator Architecture')
    return group

def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    # fmt: off
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--update-freq', default=1, type=int, 
                       help='update parameters every N batches')
    group.add_argument('--G_lr', '--G_learning-rate', default='0.25', type=float,#eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--D_lr', '--D_learning-rate', default='0.25', type=float,#eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    """
    group.add_argument('--r1-gamma', default=10.0, type=float, metavar='G')
    group.add_argument('--r2-gamma', default=0.0, type=float, metavar='G')
    group.add_argument('--critic-iter', default=5, type=int, metavar='C')
    group.add_argument('--pixel-loss', default=True, type=bool, metavar='CRI')
    group.add_argument('--pixel-loss-type', default='l1', type=str, metavar='CRI')  
    group.add_argument('--pixel-loss-weight', default=1.0, type=float, metavar='CRI')
    """
    # fmt: on
    return group

def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    # fmt: off
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--save-log-dir', metavar='DIR', default='log',
                       help='path to save log')
    #group.add_argument('--save-valid-images', metavar='DIR', default='valid_images',
    #                   help='path to save images')
    group.add_argument('--save-top-k', type=int, default=1, metavar='K',
                       help='keep best k epoch checkpoints')
    # fmt: on
    return group

def parse_args_and_arch(parser, input_args=None, parse_known=False):
    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, 'model'):
        model_specific_group = parser.add_argument_group(
            'Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        GAN_ARCH_MODEL_REGISTRY[args.model].add_args(model_specific_group)

    if hasattr(args, 'G_arch'):
        G_model_specific_group = parser.add_argument_group(
            'Generator_Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        GENERATOR_ARCH_MODEL_REGISTRY[args.G_arch].add_args(G_model_specific_group)

    if hasattr(args, 'D_arch'):
        D_model_specific_group = parser.add_argument_group(
            'Discriminator_Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        DISCRIMINATOR_ARCH_MODEL_REGISTRY[args.D_arch].add_args(D_model_specific_group)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Apply architecture configuration.
    if hasattr(args, 'model'):
        GAN_ARCH_CONFIG_REGISTRY[args.model](args)
    if hasattr(args, 'G_arch'):
        GENERATOR_ARCH_CONFIG_REGISTRY[args.G_arch](args)
    if hasattr(args, 'D_arch'):
        DISCRIMINATOR_ARCH_CONFIG_REGISTRY[args.D_arch](args)

    if parse_known:
        return args, extra
    else:
        return args

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]
