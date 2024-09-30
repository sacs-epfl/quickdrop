import sys
import argparse

from torch.fft import Tensor

sys.path.append('../*')

from env_generator.preprocessing.baselines_dataloader import *
from env_generator.utils import *
from utils.fed_utils import *

FIXED_PREFIX = 'dilichlet'
FIXED_SAVING_FORMAT = '{}-seed{}-u{}-alpha{}'  # dataset name - seed - num_client - alpha


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # print(label_distribution)
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    return client_idcs


def craft_train(d, clz_split_index):
    X_tmp, y_tmp = [], []
    for clz_index in clz_split_index:
        for i in clz_index:
            X_tmp.append(d[i][0].tolist())
            if type(d[i][1]) is Tensor:
                y_tmp.append(d[i][1].tolist())
            else:
                y_tmp.append(d[i][1])
    return X_tmp, y_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    parser.add_argument('--num_clients', type=int, default=10, help='The number of clients')
    parser.add_argument('--alpha', type=float, default=0.1, help='Lower alpha indicates higher non-iid')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    args = parser.parse_args()

    data_type = args.dataset_name
    setup_seed(args.seed)
    torch.manual_seed(args.seed)
    train_data, test_data, len_classes = load_data(data_type, download=True, save_pre_data=False)
    writing_name = data_type.lower()
    # input_sz, num_cls = train_data.data[0].shape, len_classes
    # print(input_sz, num_cls)
    N_CLIENTS = args.num_clients
    DIRICHLET_ALPHA = args.alpha
    root = '../../env/{}/{}/{}'.format(FIXED_PREFIX, data_type, FIXED_SAVING_FORMAT.format(data_type.lower(), args.seed, N_CLIENTS, args.alpha))
    if not os.path.exists(root):
        os.makedirs(root)
    train_root = '{}/{}'.format(root, 'train')
    test_root = '{}/{}'.format(root, 'test')
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    train_path = 'train.pt'
    test_path = 'test.pt'
    if args.dataset_name == 'SVHN':
        train_labels = np.array(train_data.labels, dtype=int)
    else:
        train_labels = np.array(train_data.targets, dtype=int)
    # print(train_labels)
    train_images = np.array(train_data.data)
    client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
    train_dataset = {'users': [], 'user_data': {}, 'num_samples': []}
    Xs, ys = {}, {}
    for i in range(N_CLIENTS):
        Xs[i], ys[i] = craft_train(train_data, client_idcs[i])
    for i in tqdm(range(N_CLIENTS)):
        uname = 'f_{0:05d}'.format(i)
        train_dataset['users'].append(uname)
        # print(ys[i])
        train_dataset['user_data'][uname] = {
            'x': torch.tensor(Xs[i], dtype=torch.float32),
            'y': torch.tensor(ys[i], dtype=torch.int64)}
        train_dataset['num_samples'].append(len([item for l in client_idcs[i] for item in l]))
    print(f"Clients: {train_dataset['users']}")
    print(f"Allocation result: {train_dataset['num_samples']}")
    train_save_path = '{}/{}'.format(train_root, train_path)
    print('save to {}'.format(train_save_path))
    torch.save(train_dataset, train_save_path)
