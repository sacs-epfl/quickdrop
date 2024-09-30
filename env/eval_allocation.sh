#!/bin/bash
# activate the conda environment
conda activate MW24Fall-QuickDrop-Official

# Run the Python script
python eval_allocation.py

# deactivate the conda environment to avoid any unpleasant change
conda deactivate