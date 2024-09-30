#!/bin/bash
# activate the conda environment
conda init
conda activate MW24Fall-QuickDrop-Official

# Run the Python script
python dilichelet_allocate/dilichlet_allocator.py --dataset_name CIFAR10 --num_clients 20 --alpha 10.0 --seed 7

read -p "Press enter to continue"

# deactivate the conda environment to avoid any unpleasant change
conda deactivate
