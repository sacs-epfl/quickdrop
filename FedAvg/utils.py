import torch
import numpy as np
import time
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def organise_by_class(num_classes: int, data, device):
    indices_class = [[] for c in range(num_classes)]
    images = [torch.unsqueeze(data[i][0], dim=0) for i in range(len(data))]
    labels = [data[i][1] for i in range(len(data))]
    for i, lab in enumerate(labels):
        indices_class[lab].append(i)
    images = torch.cat(images, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return indices_class, images, labels