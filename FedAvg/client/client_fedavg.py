import copy

import torch.optim

from utils.fed_utils import *
from utils.optimizers import *
from torch.utils.data import DataLoader


class FedAvgClient:
    def __init__(self,
                 args,
                 id,
                 device,
                 trainset,
                 testset,
                 class_testsets):
        self.id = id
        self.args = args
        self.trainset = trainset
        self.testset = testset
        # print(f'{id}: {len(trainset)}')
        self.train_dataloader = DataLoader(trainset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=self.args.num_workers,
                                           pin_memory=self.args.pin_memory,
                                           persistent_workers=self.args.persistent_workers)
        self.test_dataloader = self.args.test_loader
        self.class_test_dataloaders = self.args.class_testloaders

        # train parameters:
        self.device = device
        self.local_model = None
        self.epoch = None

        self.weight = 1

        # extra parameters:
        self._loss_fc = nn.CrossEntropyLoss()
        self.accuracy_records = []          # size: communication round * local epoch
        self.class_accuracy_records = []    # size: communication round * class_num
        self.loss_records = []              # size: communication round * local epoch
        self.update_records = []            # size: communication round

    # local update
    def local_update(self, epoch, model, lr, weight_decay, momentum):
        print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        per_epoch_accuracy_records = []

        # download global model:
        self.local_model = copy.deepcopy(model)
        # prepare training:
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # loss_fc = nn.CrossEntropyLoss()

        for i in range(epoch):
            loss_fc = nn.CrossEntropyLoss()
            train_loop(self.device, self.train_dataloader, self.local_model, loss_fc, optimizer)
            acc, loss = test_loop(self.device, self.test_dataloader, self.local_model, loss_fc)
            per_epoch_accuracy_records.append(acc)
        self.accuracy_records.append(per_epoch_accuracy_records)
        self.class_accuracy_records.append(self.test_class_accuracy())
        print('{} {}'.format(get_time(), 'Client {} completes update: acc {}, loss {}'.format(self.id, acc, loss)))

    def weighted_local_update(self, epoch, model, lr, weight_decay, momentum):
        print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        per_epoch_accuracy_records = []

        # download global model:
        self.local_model = copy.deepcopy(model)
        # prepare training:
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # loss_fc = nn.CrossEntropyLoss()

        for i in range(epoch):
            loss_fc = nn.CrossEntropyLoss()
            self.weight = train_loop(self.device, self.train_dataloader, self.local_model, loss_fc, optimizer)
            acc, loss = test_loop(self.device, self.test_dataloader, self.local_model, loss_fc)
            per_epoch_accuracy_records.append(acc)
        self.accuracy_records.append(per_epoch_accuracy_records)
        self.class_accuracy_records.append(self.test_class_accuracy())
        print('{} {}'.format(get_time(), 'Client {} completes update: acc {}, loss {}'.format(self.id, acc, loss)))

    def test_class_accuracy(self):
        return [test_loop(self.device, self.class_test_dataloaders[c], self.local_model, self._loss_fc)[0] for c in self.class_test_dataloaders.keys()]

    def to_disk(self):
        root = '../save/tseed{}-{}-client-avg'.format(self.args.seed, self.args.env)
        if not os.path.exists(root):
            os.mkdir(root)
        # name = 'tseed{}-{}-client-{}-accuracy-record.csv'.format(self.args.seed, self.args.env, self.id)
        name = 'tseed{}-client-{}-accuracy-record.csv'.format(self.args.seed, self.id)
        name = os.path.join(root, name)
        np.savetxt(name, tuple(itertools.chain.from_iterable(self.accuracy_records)), delimiter=",", fmt='% s')
        root = '../save/tseed{}-{}-client-class'.format(self.args.seed, self.args.env)
        if not os.path.exists(root):
            os.mkdir(root)
        # name = 'tseed{}-{}-client-{}-class-accuracy-record.csv'.format(self.args.seed, self.args.env, self.id)
        name = 'tseed{}-client-{}-class-accuracy-record.csv'.format(self.args.seed, self.id)
        name = os.path.join(root, name)
        np.savetxt(name, self.class_accuracy_records, delimiter=",", fmt='% s')

