from FedSGA.client.client_fedsga import *


class FedQuickDropClient(FedSGAClient):
    def __init__(self,
                 args,
                 id,
                 device,
                 trainset,
                 testset,
                 class_testsets):
        super().__init__(args, id, device, trainset, testset, class_testsets)
        self.args.dsa_param = ParamDiffAug()
        self.args.dsa = True if self.args.method == 'DSA' else False
        self.image_syn_train = None
        self.label_syn_train = None

    def local_update_with_affine_dataset(self, local_epoch, model, weighted=False):
        print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        per_epoch_accuracy_records = []
        ''' download global model '''
        self.local_model = copy.deepcopy(model)

        ''' indicate the desired dataset '''
        dst_train = self.trainset
        ''' organize the dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(self.args.num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.args.device)

        self.weight = len(labels_all) if weighted else 1

        ''' extraction logics '''
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        ''' quantity of distilled images '''
        ipcs = [math.ceil(len(indices_class[c]) * self.args.scale) for c in range(self.args.num_classes)]

        ''' initialize the synthetic data '''
        print(f'\t{get_time()} initialize {int(sum(ipcs))} synthetic data')
        image_syn = torch.randn(size=(sum(ipcs), self.args.channel, self.args.im_size[0], self.args.im_size[1]), dtype=torch.float, requires_grad=True, device=self.args.device)
        label_syn = np.concatenate([np.ones(ipcs[c]) * c for c in range(self.args.num_classes) if ipcs[c] != 0])
        label_syn = torch.tensor(label_syn, dtype=torch.long, requires_grad=False, device=self.args.device).view(-1)
        # print(label_syn)

        if self.args.init == 'real':
            # print('\tinitialize synthetic data from random real images')
            # override the random noise
            for c in range(self.args.num_classes): image_syn.data[sum(ipcs[:c]):sum(ipcs[:c + 1])] = get_images(c, ipcs[c]).detach().data
        else:
            # print('\tinitialize synthetic data from random noise')
            pass

        ''' prepare for training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.args.lr_img, momentum=0.5, weight_decay=self.args.weight_decay)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        net_parameters = list(self.local_model.parameters())
        optimizer_net = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr_net, weight_decay=self.args.weight_decay)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        self.args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

        ''' train synthetic data '''
        for ol in range(local_epoch):
            ''' freeze the running mu and sigma for BatchNorm layers '''
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the training with BatchNorm layers easier.
            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in self.local_model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(self.args.num_classes)], dim=0)
                self.local_model.train()  # for updating the mu, sigma of BatchNorm
                output_real = self.local_model(img_real)  # get running mu, sigma
                for module in self.local_model.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        module.eval()  # fix mu and sigma of every BatchNorm layer
            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(self.args.device)
            accumulated_grads = [torch.zeros_like(param) for param in net_parameters]
            for c in range(self.args.num_classes):  # c == il
                if ipcs[c] == 0:
                    continue
                if ipcs[c] == 1 and len(indices_class[c]) == 1:
                    continue
                img_real = get_images(c, self.args.batch_real if self.args.batch_real >= ipcs[c] else ipcs[c])
                lab_real = torch.ones((img_real.shape[0],), device=self.args.device, dtype=torch.long) * c
                img_syn = image_syn[sum(ipcs[:c]): sum(ipcs[:c + 1])].reshape((ipcs[c], self.args.channel, self.args.im_size[0], self.args.im_size[1]))
                lab_syn = torch.ones((ipcs[c],), device=self.args.device, dtype=torch.long) * c
                if self.args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, self.args.dsa_strategy, seed=seed, param=self.args.dsa_param)
                    img_syn = DiffAugment(img_syn, self.args.dsa_strategy, seed=seed, param=self.args.dsa_param)
                output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                output_syn = self.local_model(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True, retain_graph=True)
                if self.args.directly_update:
                    accumulated_grads = [accum_grad + g for accum_grad, g in zip(accumulated_grads, gw_real)]
                loss += match_loss(gw_syn, gw_real, self.args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                label_syn.detach())  # avoid any unaware modification
            self.image_syn_train = image_syn_train
            self.label_syn_train = label_syn_train

            if self.args.directly_update:
                with torch.no_grad():  # We do not want to track operations here
                    for param, grad in zip(self.local_model.parameters(), accumulated_grads):
                        # grad += self.args.weight_decay * param # Apply weight decay
                        param -= self.args.lr_net * grad  # Apply gradient descent update
            else:
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, batch_size=self.args.batch_train, shuffle=True,
                                         num_workers=self.args.num_workers)
                for il in range(local_epoch):
                    epoch('train', trainloader, self.local_model, optimizer_net, criterion, self.args, aug=True if self.args.dsa else False)

            acc, loss = test_loop(self.device, self.test_dataloader, self.local_model, self._loss_fc)
            per_epoch_accuracy_records.append(acc)
        self.accuracy_records.append(per_epoch_accuracy_records)
        self.class_accuracy_records.append(self.test_class_accuracy())
        print('{} {}'.format(get_time(), 'Client {} completes update: acc {}, loss {}'.format(self.id, acc, loss)))