from FedNaiveRetrain.client.client_fednaiveretrain import *
from FedSGA.utils import *


class FedSGAClient(FedNaiveRetrainClient):
    def __init__(self,
                 args,
                 id,
                 device,
                 trainset,
                 testset,
                 class_testsets):
        super().__init__(args, id, device, trainset, testset, class_testsets)
        self.unlearnset = None
        self.unlearn_loader = None

    def craft_class_unlearnset(self, clz):
        self.unlearnset = self.class_trainset[clz]
        self.unlearn_loader = DataLoader(self.unlearnset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def sga(self, model, forgetting_epoch, forgetting_rate, weight_decay):
        per_epoch_accuracy_records = []
        # download global model
        self.local_model = copy.deepcopy(model)

        # craft sga optimizer:
        forgetting_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=forgetting_rate, weight_decay=weight_decay)
        sum_time = 0
        for fr in range(forgetting_epoch):
            start_time = time.time()
            unlearn(self.device, self.unlearn_loader, self.local_model, self._loss_fc, forgetting_optimizer)
            end_time = time.time()
            sum_time += end_time - start_time
            acc, loss = test_loop(self.device, self.test_dataloader, self.local_model, self._loss_fc)
            per_epoch_class_accuracy = self.test_class_accuracy()
            self.class_accuracy_records.append(per_epoch_class_accuracy)
            print(f'{get_time()} Client {self.id} conducts sga and its class accuracy is {per_epoch_class_accuracy}, avg acc {acc}')
            per_epoch_accuracy_records.append(acc)
            # print(f'{get_time()} Client {self.id} forgetting epoch {fr} in {end_time - start_time}')
        self.accuracy_records.append(per_epoch_accuracy_records)
        self.class_accuracy_records.append(self.test_class_accuracy())
        # print(f'{get_time()} Client {self.id} forgetting epoch sum in {sum_time}')
