import os
from subprocess import check_output

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import options
import models


def save_load_checkpoint(args):
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=args.save_dir,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.monitor,
        mode=args.mode,
        prefix=''
        )
    return checkpoint_callback

def early_stop(args):
    if args.patience :
        early_stop_callback = EarlyStopping(
            monitor=args.monitor,
            min_delta=0.00,
            patience=args.patience,
            verbose=True,
            mode=args.mode,
        )
    else:
        early_stop_callback = None
    return early_stop_callback


def main():
    #parser = options.get_training_parser()
    parser = options.get_all_parser()
    args = options.parse_args_and_arch(parser)
    torch.manual_seed(args.seed)
   
    # Saving log and images
    os.makedirs(args.save_log_dir, exist_ok=True)
    #os.makedirs(args.save_valid_images, exist_ok=True)

    model = models.build_gan(args)
    trainer = pl.Trainer(show_progress_bar=args.progress_bar,
                         checkpoint_callback=save_load_checkpoint(args),
                         early_stop_callback=early_stop(args),
                         default_save_path=args.save_log_dir,
                         gpus=args.num_gpu, distributed_backend='dp',
                         train_percent_check=args.train_subset_split,
                         accumulate_grad_batches=args.update_freq,
                        )
    trainer.fit(model)

if __name__ == '__main__':
    main()
