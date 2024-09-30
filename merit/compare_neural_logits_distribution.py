import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.fed_utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)
    return np.array(all_losses)


def compute_logits_for_neural(net, loader):
    # TODO
    all_logits = []
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        batch_logits = net(inputs)
        for l in batch_logits:
            all_logits.append(np.array(l.detach().cpu()))
    return np.array(all_logits)


def organise_by_class(num_classes: int, data, device):
    indices_class = [[] for c in range(num_classes)]
    images = [torch.unsqueeze(data[i][0], dim=0) for i in range(len(data))]
    labels = [data[i][1] for i in range(len(data))]
    for i, lab in enumerate(labels):
        indices_class[lab].append(i)
    images = torch.cat(images, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return indices_class, images, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # parser.add_argument('--m1', type=str,
    #                     default='../FedNaiveRetrain/check_point/naive-retrain-CIFAR10-ConvNet-cifar10-seed42-u20-alpha0.1-cr50-le5.pth',
    #                     help='model')
    # parser.add_argument('--m1_label', type=str,
    #                     default='Full Set Retrain 0-8',
    #                     help='m1 name')
    parser.add_argument('--m1', type=str,
                        default='../FedSGA/check_point/FS-CIFAR10-ConvNet-cifar10-seed42-u20-alpha0.1-cr100-le5.pth',
                        help='model')
    parser.add_argument('--m1_label', type=str,
                        default='Full Set Train 0-9',
                        help='m1 name')

    # parser.add_argument('--m2', type=str,
    #                     default='../FedRecover/check_point/recover-CIFAR10-ConvNet-affine-cifar10-seed42-u20-alpha0.1-cr30-le5.pth',
    #                     help='model')
    # parser.add_argument('--m2_label', type=str,
    #                     default='Quickdrop SGA 9',
    #                     help='m2 name')

    parser.add_argument('--m2', type=str,
                        default='../FedSGA/check_point/QD-CIFAR10-ConvNet-cifar10-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-le5.pth',
                        help='model')
    parser.add_argument('--m2_label', type=str,
                        default='Quickdrop Train 0-9',
                        help='m2 name')

    parser.add_argument('--env', type=str, default='../env/dilichlet/cifar10-seed42-u20-alpha0.1', help='environment')
    args = parser.parse_args()

    print(f'{get_time()} Preparing data..')
    # load the properties of original dataset
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, data_path='../data')
    # TD_dst_test = TensorDataset(dst_test.data, dst_test.targets)
    # print(dst_test.targets)

    # load the desired models
    m1 = torch.load(args.m1)
    m2 = torch.load(args.m2)

    # load the split dataset
    if not os.path.exists(args.env):
        print(f'Cannot loading env from {args.env}')
    FIX_SUFFIX = 'train/train.pt'
    env = torch.load('{}/{}'.format(args.env, FIX_SUFFIX))
    clients = env['users']
    client_data = env['user_data']
    num_samples = env['num_samples']

    # load data of a specific client:
    unlearning_client = clients[-1]
    unlearning_class = [9]
    unlearning_client_data = client_data[unlearning_client]
    unlearning_client_trainset = TensorDataset(unlearning_client_data['x'], unlearning_client_data['y'])
    remaining_class = [i for i in range(num_classes) if i not in unlearning_class]
    train_loader = DataLoader(unlearning_client_trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dst_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_testsets = {}
    class_testloaders = {}
    test_indices_class, test_images, test_labels = organise_by_class(num_classes, dst_test, DEVICE)
    for c in range(num_classes):
        print(f'\t{get_time()} Construct class {c} testloader with size {len(test_indices_class[c])}')
        img = test_images[np.array(test_indices_class[c])]
        lab = torch.ones((img.shape[0],), device=DEVICE, dtype=torch.long) * c
        class_testsets[c] = TensorDataset(img, lab)
        class_testloaders[c] = DataLoader(class_testsets[c], batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'\t{get_time()}Compute train logits for {len(unlearning_client_trainset)} samples')
    m1_train_logits = compute_logits_for_neural(m1, train_loader)
    m2_train_logits = compute_logits_for_neural(m2, train_loader)
    print(f'\t{get_time()}Compute test logits for {len(dst_test)} samples')
    m1_test_logits = compute_logits_for_neural(m1, test_loader)
    m2_test_logits = compute_logits_for_neural(m2, test_loader)

    print(m1_train_logits.shape, m2_train_logits.shape)
    print(m1_test_logits.shape, m2_test_logits.shape)

    print(m1_train_logits.shape, m2_train_logits.shape)
    print(m1_test_logits.shape, m2_test_logits.shape)

    # On full train set
    # sns.histplot(m1_train_logits, color="skyblue", label="After Quickdrop", kde=True)
    # sns.histplot(m2_train_logits, color="red", label="Before Quickdrop", kde=True)
    # plt.legend()
    # plt.savefig('Logtis Diff on Training Set.png')

    # On full test set:
    # sns.histplot(m1_test_logits, color="skyblue", label="After Quickdrop", kde=True)
    # sns.histplot(m2_test_logits, color="red", label="Before Quickdrop", kde=True)
    # plt.legend()
    # plt.savefig('Logtis Diff on Testing Set.png')

    # On class test sets:
    # m1_class_level_logits_on_test = {}
    # m2_class_level_logits_on_test = {}
    # for clz in range(num_classes):
    #     m1_class_level_logits_on_test[clz] = compute_logits_for_neural(m1, class_testloaders[clz])
    #     m2_class_level_logits_on_test[clz] = compute_logits_for_neural(m2, class_testloaders[clz])
    #     sns.histplot(m1_class_level_logits_on_test[clz], color="skyblue", label=args.m1_label, kde=True)
    #     sns.histplot(m2_class_level_logits_on_test[clz], color="red", label=args.m2_label, kde=True)
    #     plt.legend()
    #     plt.savefig(f'Logtis Diff on Class {clz} Testing Set.png')
    #     plt.cla()
    #
    # m1_remain_test_logits = np.hstack([m1_class_level_logits_on_test[clz] for clz in remaining_class])
    # m2_remain_test_logits = np.hstack([m2_class_level_logits_on_test[clz] for clz in remaining_class])
    # m1_unlearn_test_logits = np.hstack([m1_class_level_logits_on_test[clz] for clz in unlearning_class])
    # m2_unlearn_test_logits = np.hstack([m2_class_level_logits_on_test[clz] for clz in unlearning_class])
    #
    # print(m1_remain_test_logits.shape)
    # print(m2_remain_test_logits.shape)
    # print(m1_unlearn_test_logits.shape)
    # print(m2_unlearn_test_logits.shape)

    # On retaining set and unlearning training set:
    trainset_unlearning_class_idx = None
    for clz in unlearning_class:
        if trainset_unlearning_class_idx == None:
            trainset_unlearning_class_idx = unlearning_client_trainset.labels == clz
        else:
            trainset_unlearning_class_idx = trainset_unlearning_class_idx | unlearning_client_trainset.labels == clz

    trainset_remaining_class_idx = None
    for clz in remaining_class:
        if trainset_remaining_class_idx == None:
            trainset_remaining_class_idx = unlearning_client_trainset.labels == clz
        else:
            trainset_remaining_class_idx = trainset_remaining_class_idx | unlearning_client_trainset.labels == clz

    unlearn_train_img = unlearning_client_trainset[trainset_unlearning_class_idx][0]
    unlearn_train_lab = unlearning_client_trainset[trainset_unlearning_class_idx][1]
    remain_train_img = unlearning_client_trainset[trainset_remaining_class_idx][0]
    remain_train_lab = unlearning_client_trainset[trainset_remaining_class_idx][1]
    unlearn_train_set = TensorDataset(unlearn_train_img, unlearn_train_lab)
    remain_train_set = TensorDataset(remain_train_img, remain_train_lab)
    unlearn_train_dataloader = DataLoader(unlearn_train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    remain_train_dataloader = DataLoader(remain_train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # unlearn_test_img = TD_dst_test[testset_unlearning_class_idx][0]
    # unlearn_test_lab = TD_dst_test[testset_unlearning_class_idx][1]
    # remain_test_img = TD_dst_test[testset_remaining_class_idx][0]
    # remain_test_lab = TD_dst_test[testset_remaining_class_idx][1]
    # unlearn_test_set = TensorDataset(unlearn_test_img, unlearn_test_lab)
    # remain_test_set = TensorDataset(remain_test_img, remain_test_lab)
    # unlearn_test_dataloader = DataLoader(unlearn_test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # remain_test_dataloader = DataLoader(remain_test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ## Training:
    # Unlearning logits:
    m1_unlearn_train_logits = compute_logits_for_neural(m1, unlearn_train_dataloader)
    m2_unlearn_train_logits = compute_logits_for_neural(m2, unlearn_train_dataloader)
    num_neural = m1_unlearn_train_logits.shape[1]
    for neural_index in range(num_neural):
        sns.histplot(m1_unlearn_train_logits[:, neural_index], color="skyblue", label=args.m1_label, kde=True)
        sns.histplot(m2_unlearn_train_logits[:, neural_index], color="red", label=args.m2_label, kde=True)
        plt.legend()
        plt.xlim(-10, 10)
        plt.savefig(f'Neural {neural_index} Logtis Diff on Unlearning Training Set.png')
        plt.cla()

    # Retaining logits:
    m1_remain_train_logits = compute_logits_for_neural(m1, remain_train_dataloader)
    m2_remain_train_logits = compute_logits_for_neural(m2, remain_train_dataloader)
    num_neural = m1_remain_train_logits.shape[1]
    neural_index = 0
    for neural_index in range(num_neural):
        sns.histplot(m1_unlearn_train_logits[:, neural_index], color="skyblue", label=args.m1_label, kde=True)
        sns.histplot(m2_unlearn_train_logits[:, neural_index], color="red", label=args.m2_label, kde=True)
        plt.legend()
        plt.xlim(-10, 10)
        plt.savefig(f'Neural {neural_index} Logtis Diff on Retaining Training Set.png')
        plt.cla()

    ## Testing:
    # Unlearning logits:
    # sns.histplot(m1_unlearn_test_logits, color="skyblue", label="Naive Full Set Retrain", kde=True)
    # sns.histplot(m1_unlearn_test_logits, color="skyblue", label=args.m1_label, kde=True)
    # sns.histplot(m2_unlearn_test_logits, color="red", label=args.m2_label, kde=True)
    # plt.legend()
    # plt.savefig('Logtis Diff on Unlearning Testing Set.png')
    # plt.cla()

    # Retaining logits:
    # sns.histplot(m1_remain_test_logits, color="skyblue", label="Naive Full Set Retrain", kde=True)
    # sns.histplot(m1_remain_test_logits, color="skyblue", label=args.m1_label, kde=True)
    # sns.histplot(m2_remain_test_logits, color="red", label=args.m2_label, kde=True)
    # plt.legend()
    # plt.savefig('Logtis Diff on Retaining Testing Set.png')
    # plt.cla()

