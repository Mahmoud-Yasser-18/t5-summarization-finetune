#!/bin/sh
bash ./requirements.sh
cd ./One-GPU/ 
deepspeed ./train.py --deepspeed ./deepspeed-zero3-one-gpu.json

