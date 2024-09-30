import argparse
from FedAvg.client.client_fedavg import *
from utils.fed_utils import *
from env_generator.utils import *
from torch.utils.data import DataLoader
from FedAvg.utils import *


class FedAvgServer:
    def __init__(self, args):
        self.args = args
        # load scenario
        self.device = args.device
        self.args.device = self.device
        self.dataset = args.dataset

        # load properties of original dataset
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, data_path=args.data_path)
        self.channel = channel
        self.im_size = im_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.mean = mean
        self.std = std
        self.dst_train = dst_train
        self.dst_test = dst_test
        self.class_testsets = {}

        # write statistics back
        args.channel = channel
        args.im_size = im_size
        args.num_classes = num_classes
        args.class_names = class_names
        args.mean = mean
        args.std = std

        # initialize the global model
        self.model = args.model
        self.global_model = init_model(args.model, channel, num_classes, im_size).to(self.device)  # get a random model
        self.global_model.to(self.device)

        # initialize the fed env
        self.strategy = args.strategy
        self.env = args.env
        self.fed_env = init_env(args.strategy, args.env, args.env_path)

        if os.path.exists(self.fed_env):
            print('{} {}'.format(get_time(), 'Find fl env: {}'.format(self.fed_env)))
        else:
            raise Exception('{} No such fl env at: {}'.format(get_time(), self.fed_env))

        # initialize the clients
        self.users = []
        self.bz = args.batch_size
        self.user_trainsets = {}
        self.load_dataset()
        self.construct_testset_for_each_class()
        self.communication_round = args.communication_round
        self.test_loader = DataLoader(dst_test,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=self.args.num_workers,
                                      pin_memory=self.args.pin_memory,
                                      persistent_workers=self.args.persistent_workers)
        self.class_testloaders = {c: DataLoader(class_testset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=self.args.num_workers,
                                                pin_memory=self.args.pin_memory,
                                                persistent_workers=self.args.persistent_workers) for c, class_testset in
                                  self.class_testsets.items()}
        self.args.test_loader = self.test_loader
        self.args.class_testloaders = self.class_testloaders
        self.client_instances = {uid: FedAvgClient(self.args, uid, self.device, self.user_trainsets[uid], self.dst_test, self.class_testsets) for uid in self.users}
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.local_epoch = args.local_epoch

        # records
        self._loss_fc = nn.CrossEntropyLoss()   # for testing
        self.global_model_accuracy_records = {}
        self.global_model_loss_records = {}
        self.global_model_class_accuracy_records = {}

    # Global update
    def global_update(self, n=-1):
        time_cnt = 0
        for r in range(1, self.communication_round + 1):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users
            time_start = time.time()
            self.execute_update(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)
            avg_state_dict = self.aggregate(participants)
            self.update_global_model(avg_state_dict)
            self.global_model_accuracy_records[r] = self.test_global_model()
            self.global_model_class_accuracy_records[r] = self.test_global_model_class_accuracy()
        self.all_to_disk()
        print(f'{get_time()} Global Update Finished Using {time_cnt}')

    def weighted_global_update(self, n=-1):
        time_cnt = 0
        for r in range(1, self.communication_round + 1):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users
            time_start = time.time()
            self.execute_weighted_update(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)
            avg_state_dict = self.weighted_aggregate(participants)
            self.update_global_model(avg_state_dict)
            self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
            self.global_model_class_accuracy_records[len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.all_to_disk()
        print(f'{get_time()} Global Update Finished Using {time_cnt}')

    def load_check_point(self, ckpt: str):
        self.global_model.load_state_dict(torch.load(ckpt, weights_only=False).state_dict())

    def construct_testset_for_each_class(self):
        self.test_indices_class, self.test_images, self.test_labels = organise_by_class(self.num_classes, self.dst_test, self.device)
        for c in range(self.num_classes):
            print(f'\t{get_time()} Construct class {c} test set with size {len(self.test_indices_class[c])}')
            img = self.test_images[np.array(self.test_indices_class[c])]
            lab = torch.ones((img.shape[0],), device=self.device, dtype=torch.long) * c
            self.class_testsets[c] = TensorDataset(img, lab)

    def test_global_model_class_accuracy(self):
        return [test_loop(self.device, self.class_testloaders[c], self.global_model, self._loss_fc)[0] for c in range(self.num_classes)]

    def test_global_model(self):
        acc, loss = test_loop(self.device, self.test_loader, self.global_model, self._loss_fc)
        return acc, loss

    def update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def execute_update(self, participants: list):
        for client in participants:
            self.client_instances[client].local_update(self.local_epoch, self.global_model, self.lr, self.weight_decay, self.momentum)

    def execute_weighted_update(self, participants: list):
        for client in participants:
            self.client_instances[client].weighted_local_update(self.local_epoch, self.global_model, self.lr, self.weight_decay, self.momentum)

    def aggregate(self, participants: list):
        usr_state_dicts = [self.client_instances[client].local_model.state_dict() for client in participants]
        avg_state_dict = copy.deepcopy(usr_state_dicts[0])
        for key in avg_state_dict.keys():
            for i in range(1, len(participants)):
                avg_state_dict[key] += usr_state_dicts[i][key]
            avg_state_dict[key] = torch.div(avg_state_dict[key], len(usr_state_dicts))
        return avg_state_dict

    def weighted_aggregate(self, participants: list):
        usr_state_dicts = [self.client_instances[client].local_model.state_dict() for client in participants]
        weights = torch.tensor([self.client_instances[client].weight for client in participants]).to(self.device)
        print(f'{get_time()} Weighted Aggregate for {len(participants)} participants, weights: {weights}')
        weights = weights / weights.sum()
        assert len(weights) == len(usr_state_dicts)
        avg_state_dict = copy.deepcopy(usr_state_dicts[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = 0
            for i in range(0, len(participants)):
                avg_state_dict[key] += usr_state_dicts[i][key] * weights[i]
        return avg_state_dict

    # load scenario
    def load_dataset(self):
        self.load_train_dataset()

    def load_train_dataset(self):
        train_cluster = torch.load('{}/{}/{}'.format(self.fed_env, 'train', 'train.pt'))
        self.users = train_cluster['users']
        for usr in self.users:
            tmp = train_cluster['user_data'][usr]
            self.user_trainsets[usr] = TensorDataset(tmp['x'], tmp['y'])

    def global_info_to_disk(self):
        root = '../save'
        if not os.path.exists(root):
            os.mkdir(root)
        root = '../save/tseed{}-{}-global-avg'.format(self.args.seed, self.env)
        if not os.path.exists(root):
            os.mkdir(root)
        # name = 'tseed{}-avg-{}-{}-cr{}-ep{}-{}.csv'.format(self.args.seed, self.dataset, self.env, self.communication_round, self.local_epoch, self.model)
        name = 'tseed{}-avg-{}-cr{}-ep{}-{}.csv'.format(self.args.seed, self.dataset, self.communication_round, self.local_epoch, self.model)
        fullname = '{}/{}'.format(root, name)
        tmp = pandas.DataFrame(self.global_model_accuracy_records).T
        tmp.to_csv(fullname, sep=',', index=False)
        print(f'{get_time()} Save the result to {fullname}')

        root = '../save/tseed{}-{}-global-class'.format(self.args.seed, self.env)
        if not os.path.exists(root):
            os.mkdir(root)
        # name = 'tseed{}-class-{}-{}-cr{}-ep{}-{}.csv'.format(self.args.seed, self.dataset, self.env, self.communication_round, self.local_epoch, self.model)
        name = 'tseed{}-class-{}-cr{}-ep{}-{}.csv'.format(self.args.seed, self.dataset, self.communication_round, self.local_epoch, self.model)
        fullname = '{}/{}'.format(root, name)
        tmp = pandas.DataFrame(self.global_model_class_accuracy_records).T
        tmp.to_csv(fullname, sep=',', index=False)
        print(f'{get_time()} Save the result to {fullname}')

    def all_to_disk(self):
        self.global_info_to_disk()
        for client in self.client_instances.values():
            client.to_disk()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    # Device: cpu/gpu/gpu:# if you want to indicate a specific running device
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    # FL training environment:
    parser.add_argument('--data_path', type=str, default='../../data', help='dataset')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--env_path', type=str, default=None, help='environment path')
    parser.add_argument('--strategy', type=str, default='quickdrop-affine', help='strategy')
    parser.add_argument('--env', type=str, default='affine-mnist-seed42-u20-alpha0.1-scale0.05', help='FL env')
    # Training hyperparameters:
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--communication_round', type=int, default=50, help='FL communication round')
    parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Learning rate decay if applicable')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # Caution: multiprocess support - the blow settings require huge amount of cpu cores if you want to modify.
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory')
    parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers')
    args = parser.parse_args()

    seed_bag = [0]
    # init:
    for s in seed_bag:
        args.seed = s
        setup_seed(args.seed)
        server = FedAvgServer(args)
        server.global_update(8)
        torch.save(server.global_model, '../check_point/{}-{}-{}-cr{}-le{}.pth'.format(args.dataset, args.model, args.env, args.communication_round, args.local_epoch))