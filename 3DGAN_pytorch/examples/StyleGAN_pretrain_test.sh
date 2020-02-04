#!bin/bash/

SAVE_DIR=$(pwd)

python ../test.py \
    --train-or-test test \
    --test-subset /work/smt2/bliao/StyleGAN/dataset/lr2hr/small_dataset \
    --restore-checkpoint ~/Github/3DGAN_pytorch/examples/checkpoints/_ckpt_epoch_1.ckpt \
    --meta-tags ~/Github/3DGAN_pytorch/examples/lightning_logs/version_0/meta_tags.csv \
    --model stylegan_generator_pretrain \
    --G_arch stylegan_generator_noflatten \
    --D_arch stylegan_discriminator_16 \
    --img-size 256 \
    --progress-bar False \
    --num-gpu 1 \
    --num-workers 0 \
    --use-noise False \


