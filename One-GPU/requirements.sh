#!/bin/sh
sudo yum -y install gcc-c++
if [ ! -f "/cuda_11.4.0_470.42.01_linux.run" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
fi

sudo sh cuda_11.4.0_470.42.01_linux.run

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install sentencepiece
pip install datasets==1.11.0
pip install pandas==1.1.5
pip install git+https://github.com/huggingface/transformers
pip install rouge-score==0.0.4
pip install nltk==3.2.5 
pip install deepspeed==0.4.4
