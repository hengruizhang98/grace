import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers = 2):
        super(Encoder, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

        self.convs.append(GraphConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat))

        return feat

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, in_dim, bias=True)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)

class Grace(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = Encoder(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        return th.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: th.exp(x / self.temp)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()

        return -th.log(between_sim.diag() / x1)

    def get_embedding(self, graph, feat):
        h = self.encoder(graph, feat)
        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2):

        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        z1 = self.proj(h1)
        z2 = self.proj(h2)

        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()

