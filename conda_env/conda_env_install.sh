#!/bin/bash
# create a new conda environment
conda create -n MW24Fall-QuickDrop-Official
conda activate MW24Fall-QuickDrop-Official

# install packages:
conda install seaborn # for visualization
conda install tqdm # for progress visualization
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # refer here to get latest version: https://pytorch.org/get-started/locally/
conda install scripy # odd missing when open in new environment