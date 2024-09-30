from torch.utils.data import ConcatDataset

from FedAvg.client.client_fedavg import *


class FedNaiveRetrainClient(FedAvgClient):
    def __init__(self,
                 args,
                 id,
                 device,
                 trainset,
                 testset,
                 class_testsets):
        super().__init__(args, id, device, trainset, testset, class_testsets)
        self.class_trainset = {}
        self.craft_class_trainset()

    def craft_class_trainset(self):
        for clz in range(self.args.num_classes):
            if self.args.dataset == 'SVHN':
                idx = self.trainset.labels == clz
            else:
                idx = self.trainset.labels == clz
            if sum(idx) == 0:
                self.class_trainset[clz] = None
                continue
            img = copy.deepcopy(self.trainset[idx][0])
            lab = copy.deepcopy(self.trainset[idx][1])
            self.class_trainset[clz] = TensorDataset(img, lab)

    def report_availability(self, remaining):
        assert len(remaining) >= 1
        idx = self.trainset.labels == -1
        for clz in remaining:
            if self.args.dataset == 'SVHN':
                idx = (self.trainset.labels == clz) | idx
            else:
                idx = (self.trainset.labels == clz) | idx
            # print(f'\t{get_time()} Query {clz} has data #{sum(idx)}')
        return sum(idx) != 0

    def craft_retrainset(self, remaining):
        # clean old dataloader
        self.train_dataloader = None
        # create a new
        assert len(remaining) >= 1
        idx = self.trainset.labels == -1
        for clz in remaining:
            if self.args.dataset == 'SVHN':
                idx = (self.trainset.labels == clz) | idx
            else:
                idx = (self.trainset.labels == clz) | idx
            print(f'\t{get_time()} Query {clz} has data #{sum(idx)}')
        img = copy.deepcopy(self.trainset[idx][0])
        lab = copy.deepcopy(self.trainset[idx][1])
        combined_dataset = TensorDataset(img, lab)
        self.train_dataloader = DataLoader(combined_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=True,
                                           num_workers=self.args.num_workers,
                                           pin_memory=self.args.pin_memory,
                                           persistent_workers=self.args.persistent_workers)
        print(f'{get_time()} Client {self.id} construct new a train dataloader including class {remaining} and size {len(lab)}')