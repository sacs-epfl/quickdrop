import sys
sys.path.append('../*')

import os
import time
import copy
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from utils.fed_utils import get_loops, get_dataset, init_model, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug


def distillate_dataset(args, dst_train):
        args.outer_loop, args.inner_loop = get_loops(1)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        args.dsa = True if args.method == 'DSA' else False

        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        data_save = []
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(args.num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        dc_num_table = [len(indices_class[c]) * args.dc_ratio for c in range(args.num_classes)]
        real_num_table = [len(indices_class[c]) for c in range(args.num_classes)]
        for i, item in enumerate(dc_num_table):
            if item == 0:
                dc_num_table[i] = 0
            if item > 0 and item < 1:
                dc_num_table[i] = 1
            else:
                dc_num_table[i] = math.ceil(item)
        for c in range(args.num_classes):
            print('%s class c = %d: %d (%d * %s) dc images' % (get_time(), c, dc_num_table[c], len(indices_class[c]), args.dc_ratio))
        print(dc_num_table)
        # the number of distillation data for each class - 0: Not included, >1: Included and greater than 1 , =1: Included only 1

        # for c in range(args.num_classes):
        #     print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        def get_csyn_start(c, ref):
            return int(sum(ref[:c]))
        def get_csyn_end(c, ref):
            return int(sum(ref[:c + 1]))
        # for ch in range(args.channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data '''
        # images
        print(f'{get_time()} initialize {int(sum(dc_num_table))} synthetic data')
        # img_syn = []
        # for c in range(args.num_classes):
        #     print(f'{get_time()} Class {c}: number of real dataset: {real_num_table[c]}, number of dc dataset: {dc_num_table[c]}')
        #     if real_num_table[c] == 0:
        #         continue
        #     if real_num_table[c] == 1:
        #         tmp = images_all[indices_class[c][0]].reshape(1, args.channel, args.im_size[0], args.im_size[1]).clone().requires_grad_()
        #         # tmp.requires_grad=True
        #         img_syn.append(tmp)
        #         print(tmp)
        #     else:
        #         for _ in range(dc_num_table[c]):
        #             tmp = torch.randn(size=(1, args.channel, args.im_size[0], args.im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        #             img_syn.append(tmp)
        #             print(tmp.shape)
        # image_syn = torch.cat(img_syn)
        # print(image_syn.shape)
        image_syn = torch.randn(size=(sum(dc_num_table), args.channel, args.im_size[0], args.im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        print(image_syn.shape)

        # labels
        tmp = np.array([])
        for c in range(args.num_classes):
            if dc_num_table[c] == 0:
                continue
            c_label_syn = np.ones(dc_num_table[c]) * c
            if tmp.size == 0:
                tmp = c_label_syn
            else:
                tmp = np.concatenate((tmp, c_label_syn))
        label_syn = torch.tensor(tmp, dtype=torch.long, requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        print(label_syn)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            # TODO: ratio version
            for c in range(args.num_classes):
                image_syn.data[sum(dc_num_table[:c]):sum(dc_num_table[:c + 1])] = get_images(c, dc_num_table[c]).detach().data
        else:
            print('initialize synthetic data from random noise')

        # for c, num in enumerate(real_num_table):
        #     if num > 1:
        #         continue
        #     cloned_image = images_all[indices_class[c][0]].detach()
        #     img_shape = cloned_image.shape
        #     print(f'{get_time()} Overwrite synthetic image ({get_csyn_start(c, dc_num_table)}, {get_csyn_end(c, dc_num_table)}) with real image {cloned_image}')
        #     for (img_channel, img_x, img_y) in zip(range(img_shape[0]), range(img_shape[1]), range(img_shape[2])):
        #         image_syn[get_csyn_start(c, dc_num_table): get_csyn_end(c, dc_num_table)][img_channel][img_x][img_y] = cloned_image[img_channel][img_x][img_y]


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):
            ''' Train synthetic data '''
            net = init_model(args.model, args.channel, args.num_classes, args.im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
            for ol in range(args.outer_loop):
                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()  # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)  # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer
                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(args.num_classes):
                    if real_num_table[c] == 0:
                        continue
                    if real_num_table[c] == 1:
                        continue
                    img_real = get_images(c, args.batch_real if args.batch_real >= real_num_table[c] else real_num_table[c])
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[sum(dc_num_table[:c]): sum(dc_num_table[:c + 1])].reshape(
                        (dc_num_table[c], args.channel, args.im_size[0], args.im_size[1]))
                    lab_syn = torch.ones((dc_num_table[c],), device=args.device, dtype=torch.long) * c
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss += match_loss(gw_syn, gw_real, args)
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()
                if ol == args.outer_loop - 1:
                    break
                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)
            loss_avg /= (args.num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration:  # only record the final results
                image_syn_copy = copy.deepcopy(image_syn.detach().cpu())
                label_syn_copy = copy.deepcopy(label_syn.detach().cpu())
                overwritted_image_syn = []
                for index in range(image_syn_copy.shape[0]):
                    c = label_syn_copy[index]
                    if real_num_table[c] == 1:
                        tmp = images_all[indices_class[c][0]].detach().reshape(1, args.channel, args.im_size[0], args.im_size[1]).cpu()
                        print(f'{get_time()} Process {c} image: shape {tmp.shape}')
                        overwritted_image_syn.append(tmp)
                    else:
                        tmp = image_syn_copy[index].reshape(1, args.channel, args.im_size[0], args.im_size[1]).cpu()
                        print(f'{get_time()} Process {c} image: shape {tmp.shape}')
                        overwritted_image_syn.append(tmp)
                image_syn_copy = torch.cat(overwritted_image_syn)
                print(f'{get_time()} Distillation dataset shape: {image_syn_copy.shape}')
                data_save.append([image_syn_copy, label_syn_copy])
        return data_save


def plug(img, label):
    # r = []
    # for x, y in zip(img, label):
    #     r.append((x, y))
    return [(x, y) for x, y in zip(img, label)]


def distillate_scenario():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
    parser.add_argument('--data_path', type=str, default='../../data', help='dataset path')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--Iteration', type=int, default=800, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--save_path', type=str, default='../env', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--allocating_strategy', type=str, default='dilichlet', help='allocating algorithm of fl scenario')
    parser.add_argument('--env', type=str, default='svhn-seed42-u20-alpha0.1', help='allocating algorithm of fl scenario')
    parser.add_argument('--dc_ratio', type=float, default='0.001', help='For each class: #dc images = #original images * ratio')

    args = parser.parse_args()
    fixed_dc_root = '../../env/data_condensation'
    if not os.path.exists(fixed_dc_root):
        os.mkdir(fixed_dc_root)
    save_to = '{}/{}'.format(fixed_dc_root, 'compressed-{}-init-{}-{}-iter{}-ratio{}'.format(args.allocating_strategy, args.init, args.env, args.Iteration, args.dc_ratio))
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    if not os.path.exists('{}/{}'.format(save_to, 'train')):
        os.mkdir('{}/{}'.format(save_to, 'train'))
        os.mkdir('{}/{}'.format(save_to, 'test'))
    args.channel, args.im_size, args.num_classes, args.class_names, args.mean, args.std, dst_train, dst_test = get_dataset(args.dataset, args.data_path)
    del dst_train
    del dst_test
    scenario_path = '../../env/{}/{}'.format(args.allocating_strategy, args.env)
    print(os.path.exists('../../env/{}/{}'.format( args.allocating_strategy, args.env)))
    train_suffix = 'train/train.pt'
    test_suffix = 'test/test.pt'
    train_root = '{}/{}'.format(scenario_path, train_suffix)
    test_root = '{}/{}'.format(scenario_path, test_suffix)
    data = torch.load(train_root)
    users = data['users']
    user_data = data['user_data']
    print('{} - {} clients'.format(get_time(), len(users)))
    for u in users:
        print('{} Condensate {}'.format(get_time(), u))
        dist_data = distillate_dataset(args, plug(user_data[u]['x'], user_data[u]['y']))[0]
        user_data[u]['x'] = dist_data[0]
        user_data[u]['y'] = dist_data[1]
    train_save_path = '{}/{}'.format(save_to, train_suffix)
    print('save to {}'.format(train_save_path))
    torch.save(data, train_save_path)


if __name__ == '__main__':
    distillate_scenario()


