import matplotlib.pyplot as plt

from FedQuickDrop.server.server_fedquickdrop import *
from FedRecover.server.server_fedrecover import FedRecoverServer


def main(args):
    if args.with_affine_dataset:
        # Create a quickdrop server.
        sga_server = FedQuickDropServer(args)
        sga_server.load_check_point(args.checkpoint_path)
    else:
        # Create a quickdrop server and train.
        sga_server = FedQuickDropServer(args)
        sga_server.weighted_global_update()

    # additional gap
    # more_reasonable_layout(server)
    # select client 9 to unlearn class 9
    forgetting_client = list(sga_server.client_instances.keys())[-1]
    forgetting_class = list(range(sga_server.num_classes))[-1]
    print(f'{get_time()} Select client {forgetting_client} to unlearn class {forgetting_class}.')
    sga_server.sga(forgetting_client, forgetting_class)
    print(f'{get_time()} Overwrite the previous global model.')
    sga_server.update_global_model(sga_server.client_instances[forgetting_client].local_model.state_dict())
    sga_server.global_model_accuracy_records[
        len(sga_server.global_model_accuracy_records) + 1] = sga_server.test_global_model()
    sga_server.global_model_class_accuracy_records[
        len(sga_server.global_model_class_accuracy_records) + 1] = sga_server.test_global_model_class_accuracy()

    remaining = list(range(sga_server.num_classes))[:9]
    sga_server.communication_round = args.recovering_round
    print(f'f{get_time()} Naively recover on the classes {remaining}')
    # recovering_server.weighted_global_naive_retrain(remaining, 8)
    sga_server.weighted_global_naive_retrain(remaining)
    torch.save(sga_server.global_model, f'../tmp/{args.dataset}_Quickdrop_model.pth')
    # additional gap
    more_reasonable_layout(sga_server)
    sga_server.all_to_disk()

    x_values = list(sga_server.global_model_class_accuracy_records.keys())
    y_values = list(sga_server.global_model_class_accuracy_records.values())
    for i in range(10):
        plt.plot(x_values, [y[i] for y in y_values], label=f'Class {i}')
    plt.legend(loc='right')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('QuickDrop Unlearns Class 9')
    plt.savefig(f'../result/{args.dataset.lower()}_vis.png')

def solve_env():
    print(f'{get_time()} Create "tmp" folder for the intermediate results.')
    if not os.path.exists('../tmp'):
        os.makedirs('../tmp')
    print(f'{get_time()} Create "result" folder for the intermediate results.')
    if not os.path.exists('../result'):
        os.makedirs('../result')

def more_reasonable_layout(s):
    for _ in range(2):
        s.global_model_accuracy_records[len(s.global_model_accuracy_records) + 1] = s.test_global_model()
        s.global_model_class_accuracy_records[
            len(s.global_model_class_accuracy_records) + 1] = s.test_global_model_class_accuracy()
    s.all_to_disk()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    # Device: cpu/gpu/gpu:# if you want to indicate a specific running device
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    # Reproduction additional parameters:
    parser.add_argument('--with_affine_dataset', type=bool, default=True, help='the affine dataset ')
    parser.add_argument('--recovering_round', type=int, default=5, help='recover round')
    # Load the affine dataset
    parser.add_argument('--data_path', type=str, default='../../../data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--env_path', type=str, default='../../../env', help='environment path')
    parser.add_argument('--strategy', type=str, default='quickdrop-affine', help='strategy')
    parser.add_argument('--env', type=str, default='affine-cifar10-seed42-u20-alpha0.1-scale0.01', help='SGA env')
    parser.add_argument('--communication_round', type=int, default=200, help='FL communication round') # Disable if with the affine dataset
    parser.add_argument('--checkpoint_path', type=str,
                        default='../../check_point/QuickDrop-ConvNet-mnist-seed42-u20-alpha0.1-lr_net0.01-lr_img0.1-scale0.01-le5.pth',
                        help='Quickdrop Well-trained Checkpoint')
    # Training hyperparameters:
    # parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate for normal global update')
    parser.add_argument('--learning_rate', type=float, default=0.02, help='Learning rate for normal global update')
    # parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for normal global update')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Learning rate decay if applicable')
    parser.add_argument('--momentum', type=float, default=0., help='Momentum')
    parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
    # Quickdrop parameters:
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA') # Disable if with the affine dataset
    parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images') # Disable if with the affine dataset
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters') # Disable if with the affine dataset
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data') # Disable if with the affine dataset
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks') # Disable if with the affine dataset
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.') # Disable if with the affine dataset
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy') # Disable if with the affine dataset
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric') # Disable if with the affine dataset
    parser.add_argument('--scale', type=float, default='0.05', help='For each class: #dc images = #original images * ratio') # Disable if with the affine dataset
    parser.add_argument('--directly_update', type=bool, default=False, help='True: update local model via synthetic loss/False: recalculate loss on synthetic dataset') # Disable if with the affine dataset
    # Affine dataset parameters:
    parser.add_argument('--affine_path', type=str, default='quickdrop-affine', help='path to save affine results') # Disable if with the affine dataset
    # Forgetting parameters
    parser.add_argument('--forgetting_epoch', type=int, default=1, help='FL forgetting epoch')
    # parser.add_argument('--forgetting_rate', type=float, default=0.004, help='Forgetting rate')
    parser.add_argument('--forgetting_rate', type=float, default=0.004, help='Forgetting rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers') # Disable if with the affine dataset
    parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory') # Disable if with the affine dataset
    parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers') # Disable if with the affine dataset
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    setup_seed(args.seed)

    solve_env()
    main(args)
