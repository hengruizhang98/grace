import argparse
import os.path as osp
import random

from model import Grace, Encoder
from aug import aug
from dataset import load

import torch as th
import torch.nn as nn

import yaml
from yaml import SafeLoader

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dataname = 'citeseer'
    device = 'cpu'
    config = 'config.yaml'

    config = yaml.load(open(config), Loader=SafeLoader)[dataname]
    lr = config['learning_rate']
    hid_dim = config['num_hidden']
    out_dim = config['num_proj_hidden']

    num_layers = config['num_layers']
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[config['activation']]

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']

    temp = config['tau']
    epochs = config['num_epochs']
    wd = config['weight_decay']

    graph, feat, labels, num_class, train_mask, val_mask, test_mask = load(dataname)

    in_dim = feat.shape[1]

    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

    print("=== Final ===")

    # graph = graph.add_self_loop()
    graph = graph.to(device)
    feat = feat.to(device)
    embeds = model.get_embedding(graph, feat)

