from FedSGA.server.server_fedsga import *
from FedQuickDrop.client.client_fedquickdrop import *


class FedQuickDropServer(FedSGAServer):
    def __init__(self, args):
        super().__init__(args)
        self.client_instances = {
            uid: FedQuickDropClient(args, uid, self.device, self.user_trainsets[uid], self.dst_test, self.class_testsets) for
            uid in self.users}

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
            self.execute_update_with_affine_dataset(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)
            avg_state_dict = self.aggregate(participants)
            self.update_global_model(avg_state_dict)
            # self.global_model_accuracy_records[r + len(self.global_model_accuracy_records.keys())] = self.test_global_model()
            # self.global_model_class_accuracy_records[r + len(self.global_model_accuracy_records.keys())] = self.test_global_model_class_accuracy()
            self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
            self.global_model_class_accuracy_records[
                len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.all_to_disk()
        print(f'{get_time()} Global Update Finished Using {time_cnt}')

    def weighted_global_update(self, n=-1):
        time_cnt = 0
        for r in tqdm(range(1, self.communication_round + 1)):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users
            time_start = time.time()
            self.weighted_execute_update_with_affine_dataset(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)
            avg_state_dict = self.weighted_aggregate(participants)
            self.update_global_model(avg_state_dict)
            # self.global_model_accuracy_records[r + len(self.global_model_accuracy_records.keys())] = self.test_global_model()
            # self.global_model_class_accuracy_records[r + len(self.global_model_accuracy_records.keys())] = self.test_global_model_class_accuracy()
            self.global_model_accuracy_records[len(self.global_model_accuracy_records) + 1] = self.test_global_model()
            self.global_model_class_accuracy_records[
                len(self.global_model_class_accuracy_records) + 1] = self.test_global_model_class_accuracy()
        self.all_to_disk()
        print(f'{get_time()} Global Update Finished Using {time_cnt}')

    def execute_update_with_affine_dataset(self, participants: list):
        for uid in participants:
            self.client_instances[uid].local_update_with_affine_dataset(self.local_epoch, self.global_model)

    def weighted_execute_update_with_affine_dataset(self, participants: list):
        for uid in participants:
            self.client_instances[uid].local_update_with_affine_dataset(self.local_epoch, self.global_model, weighted=True)

    def save_affine_dataset(self):
        if self.args.env_path is None:
            root = '../../env/{}/affine-{}-scale{}'.format(self.args.affine_path, self.args.env, self.args.scale)
        else:
            root = '{}/{}/affine-{}-{}'.format(self.args.env_path, self.args.affine_path, self.args.env, self.args.scale)
        if not os.path.exists(root):
            os.makedirs(root)
        train_root = '{}/{}'.format(root, 'train')
        test_root = '{}/{}'.format(root, 'test')
        if not os.path.exists(train_root):
            os.makedirs(train_root)
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        train_path = 'train.pt'
        train_dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        for uid in self.users:
            train_dataset['users'].append(uid)
            # print(ys[i])
            current_client = self.client_instances[uid]
            train_dataset['user_data'][uid] = {
                'x': current_client.image_syn_train,
                'y': current_client.label_syn_train}
            train_dataset['num_samples'].append(len(current_client.label_syn_train))
        print(f"Clients: {train_dataset['users']}")
        print(f"Affine dataset result: {train_dataset['num_samples']}")
        train_save_path = '{}/{}'.format(train_root, train_path)
        print('save to {}'.format(train_save_path))
        torch.save(train_dataset, train_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    # Device: cpu/gpu/gpu:# if you want to indicate a specific running device
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--data_path', type=str, default='../../data', help='dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--env_path', type=str, default=None, help='environment path')
    parser.add_argument('--strategy', type=str, default='dilichlet', help='strategy')
    parser.add_argument('--env', type=str, default='cifar10-seed42-u20-alpha0.1', help='FL env')
    parser.add_argument('--communication_round', type=int, default=200, help='FL communication round')
    # Training hyperparameters:
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for normal global update')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Learning rate decay if applicable')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
    # Quickdrop parameters:
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--scale', type=float, default=0.01, help='For each class: #dc images = #original images * ratio')
    parser.add_argument('--directly_update', type=bool, default=False, help='True: update local model via synthetic loss/False: recalculate loss on synthetic dataset')
    # Affine dataset parameters:
    parser.add_argument('--affine_path', type=str, default='quickdrop-affine', help='path to save affine results')
    # Forgetting parameters
    parser.add_argument('--forgetting_epoch', type=int, default=0, help='FL forgetting epoch')
    parser.add_argument('--forgetting_rate', type=float, default=0., help='Forgetting rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory')
    parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    setup_seed(args.seed)
    server = FedQuickDropServer(args)
    server.weighted_global_update(8)
    torch.save(server.global_model,
               '../check_point/QuickDrop-{}-{}-lr_net{}-lr_img{}-scale{}-le{}.pth'.format(args.model, args.env,
                                                                        args.lr_net,
                                                                        args.lr_img,
                                                                        args.scale,
                                                                        args.local_epoch))
    server.save_affine_dataset()

        # Forgetting params:
        # forgetting_client = list(server.client_instances.keys())[-1]
        # forgetting_class = list(range(server.num_classes))[-1]
        # print(f'f{get_time()} SGA forgetting on {forgetting_client} {forgetting_class}')
        # server.sga(forgetting_client, forgetting_class)
        # torch.save(server.global_model,
        #            '../check_point/SGA-{}-{}-{}-fe{}-fr{}.pth'.format(args.dataset, args.model, args.env,
        #                                                               args.forgetting_epoch, args.forgetting_rate))