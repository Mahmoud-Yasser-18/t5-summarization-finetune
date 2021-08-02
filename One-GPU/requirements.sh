#!/bin/sh
sudo yum -y install gcc-c++
if [ ! -f "cuda_11.1.0_455.23.05_linux.run" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
fi
sudo sh cuda_11.1.0_455.23.05_linux.run

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install sentencepiece
pip install datasets==1.11.0
pip install pandas==1.1.5
pip install git+https://github.com/huggingface/transformers
pip install rouge-score==0.0.4
pip install nltk==3.2.5 


git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log


cd ..