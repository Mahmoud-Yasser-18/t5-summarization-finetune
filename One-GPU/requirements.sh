#!/bin/sh
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install datasets==1.11.0
pip install pandas==1.1.5
pip install git+https://github.com/huggingface/transformers
pip install rouge-score==0.0.4
pip install nltk==3.2.5 
pip install deepspeed==0.4.4
