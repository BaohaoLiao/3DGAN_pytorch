#!bin/bash/

SAVE_DIR=$(pwd)

python ../train.py \
    --train-subset /work/smt2/bliao/StyleGAN/dataset/lr2hr/kc_000008 \
    --model stylegan_noflatten \
    --G_arch stylegan_generator_noflatten \
    --D_arch stylegan_discriminator_16 \
    --img-size 16 \
    --channels 1 \
    --batch-size 64 \
    --G_lr 0.00001 \
    --D_lr 0.00001 \
    --save-dir $SAVE_DIR/checkpoints \
    --save-log-dir $SAVE_DIR \
    --save-top-k 2 \
    --monitor 'val_loss' \
    --mode 'min' \
    --patience 10 \
    --seed 200 \
    --progress-bar False \
    --num-gpu 1 \
    --num-workers 0 \
    --pretrain-generator True \
    --update-freq 5 \
    


