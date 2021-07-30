#!/bin/sh
bash ./requirements.sh
deepspeed ./train.py --deepspeed ./deepspeed-zero3-one-gpu.json
