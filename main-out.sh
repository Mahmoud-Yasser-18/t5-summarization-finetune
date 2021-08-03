#!/bin/sh
bash ./requirements.sh
cd ./One-GPU/ 
NCCL_TREE_THRESHOLD=0 CUDA_VISIBLE_DEVICES= 0 deepspeed --num_gpus=8 ./train.py --deepspeed ./deepspeed-zero3-one-gpu.json

