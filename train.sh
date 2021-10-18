#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py --dataset nabirds --pretrained_dir data/ViT-B_16.npz --data_root ../datasets --model_type ViT-B_16 --fp16 --train_batch_size 16 --eval_batch_size 32 --name sample_run