
'''
Cited from: https://github.com/rruisong/pytorch_federated_learning/blob/main/preprocessing/baselines_dataloader.py
'''
import torch
import torchvision
import torchvision.transforms as transforms
from torch import cat
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, TensorDataset
import os
import PIL
from torch.utils.data import Dataset


def load_data(name, root='../../data', download=True, save_pre_data=True):

    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'CIFAR10-LT10', 'CIFAR10-LT50', 'CIFAR10-LT100', 'CIFAR100-LT10', 'CIFAR100-LT50', 'CIFAR100-LT100',
                 'QMNIST', 'SVHN', "IMAGENET", 'CIFAR100', 'AG_NEWS', 'LT-IMAGENET']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
        print(trainset.targets)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split='byclass', download=download, transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split='byclass', download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.])])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)

    elif name == 'IMAGENET':
        train_val_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
        ])
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_val_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 62, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'IMAGENET': 200,
        'CIFAR100': 100
    }

    len_classes = len_classes_dict[name]

    return trainset, testset, len_classes
