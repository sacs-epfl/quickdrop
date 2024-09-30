import os
import csv
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def process_request(args):
    p = './{}'.format(args.strategy)
    if not os.path.exists(p):
        print('No dataset directory: {}'.format(p))
        return 0
    p = '{}/{}'.format(p, args.scenario)
    if not os.path.exists(p):
        print('No strategy directory: {}'.format(p))
        return 0
    train_r = 'train/train.pt'
    train_p = '{}/{}'.format(p, train_r)
    d = torch.load(train_p)
    print(d['users'])
    usrs = d['users']
    statistics = {}
    for u in usrs:
        statistics[u] = [0 for _ in range(args.class_num)]
    for u in usrs:
        tmp = d['user_data'][u]
        tmp_y = tmp['y']
        for i in tmp_y:
            statistics[u][i] += 1
        # print(statistics[u])
    data_array = np.array([vals for vals in statistics.values()])
    data_df = pd.DataFrame(data_array)
    sns.heatmap(data_df, cmap='Blues')
    plt.savefig('{}.png'.format(args.scenario))

    labels = set()
    for usr in d['users']:
        for y in d['user_data'][usr]['y']:
            labels.add(y.item())
    field_names = ['name']
    class_names = [f'class {c}' for c in range(args.class_num)]
    field_names.extend(class_names)
    to_disk = []
    for u in usrs:
        tmp = {'name': u}
        for i in range(args.class_num):
            tmp[class_names[i]] = statistics[u][i]
        to_disk.append(tmp)
    print(to_disk[:20])
    with open(f'{args.scenario}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(to_disk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    # parser.add_argument('--strategy', type=str, default='quickdrop-affine', help='allocation strategy')
    # parser.add_argument('--scenario', type=str, default='affine-cifar10-seed42-u10-alpha0.1-scale0.01', help='scenario name')
    parser.add_argument('--strategy', type=str, default='quickdrop-affine', help='allocation strategy')
    parser.add_argument('--scenario', type=str, default='affine-cifar10-seed42-u20-alpha0.1-scale0.01', help='scenario name')
    parser.add_argument('--class_num', type=int, default=10, help='number of classes')
    args = parser.parse_args()

    process_request(args)
