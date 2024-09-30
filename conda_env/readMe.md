We use [Anaconda](https://www.anaconda.com/download/) to manage our running environment

How to install the running environment?

```
conda create -n MW24Fall-QuickDrop-Official # create a new conda vironment
conda activate MW24Fall-QuickDrop-Official

conda install seaborn # for visualization
conda install tqdm # for progress visualization
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # refer here to get latest version: https://pytorch.org/get-started/locally/
conda install scripy # odd missing when open in new environment
```

The codes already have been encapsulated into the executable file: conda_env_install.sh.
