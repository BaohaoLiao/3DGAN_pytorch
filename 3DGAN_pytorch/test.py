import os

import pytorch_lightning as pl

import options
import models


def main():
    parser = options.get_all_parser()
    args = options.parse_args_and_arch(parser)
    #print(args)
        
    # Saving test images
    os.makedirs(args.save_test_images, exist_ok=True)
        
    model = models.build_gan(args)
    """
    model = model.load_from_metrics(
                weights_path=args.restore_checkpoint,
                tags_csv=args.meta_tags,
                map_location=None)
    """
    model = model.load_from_checkpoint(
        checkpoint_path=args.restore_checkpoint
    )
    trainer = pl.Trainer(gpus=args.num_gpu)
    trainer.test(model)

if __name__ == '__main__':
    main()
