from FedNaiveRetrain.server.server_fednaiveretrain import *

class FedRecoverServer(FedNaiveRetrainServer):
    def __init__(self, args):
        super().__init__(args)

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

    # seed_bag = [0, 1, 4, 6, 7, 8, 11, 16, 21, 24]
    seed_bag = [0]
    # init:
    for s in seed_bag:
        args.seed = s
        setup_seed(args.seed)
        server = FedRecoverServer(args)
        remaining = list(range(server.num_classes))[0:9]
        print(f'f{get_time()} Naive recover on {remaining}')
        server.load_check_point('../../FedSGA/check_point/sga-CIFAR10-ConvNet-affine-cifar10-seed42-u20-alpha0.1-fe1-fr0.1.pth')
        server.global_naive_retrain(remaining, 5)
        torch.save(server.global_model, '../check_point/recover-{}-{}-{}-cr{}-le{}.pth'.format(args.dataset, args.model, args.env,
                                                                  args.communication_round, args.local_epoch))