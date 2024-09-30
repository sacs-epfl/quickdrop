from FedAvg.server.server_fedavg import *
from FedNaiveRetrain.client.client_fednaiveretrain import *


class FedNaiveRetrainServer(FedAvgServer):
    def __init__(self, args):
        super().__init__(args)
        self.client_instances = {
            uid: FedNaiveRetrainClient(args, uid, self.device, self.user_trainsets[uid], self.dst_test, self.class_testsets) for
            uid in self.users}

    def global_naive_retrain(self, remaining, n=-1):
        self.broadcast_retrain(remaining)
        for r in range(1, self.communication_round + 1):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users
            self.execute_update(participants)
            avg_state_dict = self.aggregate(participants)
            self.update_global_model(avg_state_dict)
            self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
            self.global_model_class_accuracy_records[
                len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.all_to_disk()

    def weighted_global_naive_retrain(self, remaining, n=-1):
        self.broadcast_retrain(remaining)
        for r in range(1, self.communication_round + 1):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            available_clients = []
            for user in self.users:
                if not self.client_instances[user].report_availability(remaining):
                    continue
                available_clients.append(user)
            if n != -1:
                participants = random.sample(available_clients, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = available_clients
            self.execute_weighted_update(participants)
            avg_state_dict = self.weighted_aggregate(participants)
            self.update_global_model(avg_state_dict)
            self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
            self.global_model_class_accuracy_records[
                len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.all_to_disk()

    def broadcast_retrain(self, remaining, n=-1):
        if n != -1:
            participants = random.sample(self.users, n)
        else:
            participants = self.users
        for uid in participants:
            print(f'{get_time()} Broadcast {uid} to participate the naive retrain')
            self.client_instances[uid].craft_retrainset(remaining)


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
    parser.add_argument('--communication_round', type=int, default=100, help='FL communication round')
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
        server = FedNaiveRetrainServer(args)
        remaining = list(range(server.num_classes))[0:9]
        print(f'f{get_time()} Naive retrain/recover on {remaining}')
        server.global_naive_retrain(remaining, 8)
        torch.save(server.global_model, '../check_point/naive-retrain-{}-{}-{}-cr{}-le{}.pth'.format(args.dataset, args.model, args.env,
                                                                  args.communication_round, args.local_epoch))
