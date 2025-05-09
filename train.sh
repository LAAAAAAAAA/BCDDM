#! /bin/bash

CONFIG="config/plateau_config.json"

# Single GPU
python train.py --config $CONFIG

# Multi GPU
# NUM_GPUS=2
# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     train.py \
#     --distributed \
#     --config $CONFIG