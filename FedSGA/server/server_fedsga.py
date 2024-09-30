from FedNaiveRetrain.server.server_fednaiveretrain import *
from FedSGA.client.client_fedsga import *


class FedSGAServer(FedNaiveRetrainServer):
    def __init__(self, args):
        super().__init__(args)
        self.client_instances = {uid: FedSGAClient(args, uid, self.device, self.user_trainsets[uid], self.dst_test, self.class_testsets) for uid in self.users}

    def sga(self, uid, forgetting_class):
        self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
        self.global_model_class_accuracy_records[len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.client_instances[uid].craft_class_unlearnset(forgetting_class)
        self.client_instances[uid].sga(self.global_model, self.args.forgetting_epoch, self.args.forgetting_rate, self.args.weight_decay)
        self.all_to_disk()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    # Device: cpu/gpu/gpu:# if you want to indicate a specific running device
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    # FL training environment:
    parser.add_argument('--data_path', type=str, default='../../data', help='dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--env_path', type=str, default=None, help='environment path')
    parser.add_argument('--strategy', type=str, default='dilichlet', help='strategy')
    parser.add_argument('--env', type=str, default='cifar10-seed42-u20-alpha0.1', help='FL env')
    # Training hyperparameters:
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--communication_round', type=int, default=200, help='FL communication round')
    parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
    parser.add_argument('--forgetting_epoch', type=int, default=1, help='FL forgetting epoch')
    parser.add_argument('--forgetting_rate', type=float, default=0.1, help='Forgetting rate')
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
    for s in seed_bag[:1]:
        args.seed = s
        setup_seed(args.seed)
        server = FedSGAServer(args)
        server.load_check_point('../check_point/QD-SVHN-ConvNet-svhn-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-le5.pth')
        # Params:
        # forgetting_client = list(server.client_instances.keys())[-1]
        # forgetting_class = list(range(server.num_classes))[-1]
        # print(f'f{get_time()} SGA forgetting on {forgetting_client} {forgetting_class}')
        # server.sga(forgetting_client, forgetting_class)
        # torch.save(server.global_model, '../check_point/sga-{}-{}-{}-fe{}-fr{}.pth'.format(args.dataset, args.model, args.env, args.forgetting_epoch, args.forgetting_rate))