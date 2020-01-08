#!bin/bash/

SAVE_DIR=$(pwd)

python ../train.py \
    --train-subset /work/smt2/bliao/StyleGAN/dataset/lr2hr/kc_000008_ps \
    --save-dir $SAVE_DIR/checkpoints \
    --save-log-dir $SAVE_DIR \
    --test-subset /work/smt2/bliao/StyleGAN/dataset/lr2hr/test \
    --restore-checkpoint $SAVE_DIR/checkpoints/_ckpt_epoch_best.ckpt \
    --meta-tags $SAVE_DIR/lightning_logs/version_best/meta_tags.csv \
    --save-test-images $SAVE_DIR/test_images \
    --model stylegan \
    --G_arch stylegan_generator_noflatten_more_resolutions_progressive \
    --D_arch vgg \
    --img-size 32 \
    --channels 1 \
    --batch-size 16 \
    --G_lr 0.00001 \
    --D_lr 0.00001 \
    --save-top-k 2 \
    --monitor 'val_loss' \
    --mode 'min' \
    --patience 10 \
    --seed 200 \
    --progress-bar False \
    --num-gpu 1 \
    --num-workers 0 \
    --update-freq 1 \
    --use-noise True \
    --pixel-loss-weight 0.5
    
    


