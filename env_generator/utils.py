import os
import torch
import numpy as np
import random

'''
Default root path: ../../env 
'''

FIXED_ROOT_PATH = '../../env'
# if not os.path.exists(FIXED_ROOT_PATH):
#     os.mkdir(FIXED_ROOT_PATH)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_dataset(prefix, path, dataset):
    if not os.path.exists(FIXED_ROOT_PATH):
        os.mkdir(FIXED_ROOT_PATH)
    train_path = 'train.pt'
    test_path = 'test.pt'
    train_root = '../../env/{}/{}/train'.format(prefix, path)
    test_root = '../../env/{}/{}/test'.format(prefix, path)
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    trainset_saving_path = os.path.join(train_root, train_path)
    torch.save(dataset, trainset_saving_path)

def init_env(strategy, env, root_path=FIXED_ROOT_PATH):
    if root_path is None:
        return os.path.join(FIXED_ROOT_PATH, strategy, env)
    else:
        return os.path.join(root_path, strategy, env)

# Further preprocess -> {'x': ..., 'y': ...}
def divideXy_to_tensor(data) -> dict:
    """
    normal division with class
    :param data: dataset with item: idx(0)-x, idx(1)-y
    :return: divided dataset
    """
    tensors = {}
    Xs, ys = [], []
    for i in range(len(data)):
        Xs.append(data[i][0].tolist())
        ys.append(data[i][1].tolist())
    tensors['x'] = torch.tensor(Xs, dtype=torch.float32)
    tensors['y'] = torch.tensor(ys, dtype=torch.int64)
    return tensors


