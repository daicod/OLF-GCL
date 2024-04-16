import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from gcn import GCN
import leading_tree as lt


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        #features @ self.weight @ summary.t()
        return torch.matmul(features, torch.matmul(self.weight, summary))


class DiscriminatorLT(nn.Module):
    def __init__(self, n_hidden):
        super(DiscriminatorLT, self).__init__()

    def forward(self, features, summary):
        n, h = features.size()

        ####features =  features / features.norm(dim=1)[:, None]
        # features = torch.sum(features*summary, dim=1)

        # features = features @ self.weight @ summary.t()
        return torch.bmm(features.view(n, 1, h), summary.view(n, h, 1))  # torch.sum(features*summary, dim=1)


class OLF_GCL(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, beta, alpha):
        super(OLF_GCL, self).__init__()
        self.n_hidden = n_hidden
        self.g = g
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.discriminatorLT = DiscriminatorLT(n_hidden)

        self.loss = nn.BCEWithLogitsLoss()
        self.beta = beta

        self.alpha = alpha
        

    def forward(self, features, center_index):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        graph_summary = torch.sigmoid(positive.mean(dim=0))
        pos_graph = self.discriminator(positive, graph_summary)
        neg_graph = self.discriminator(negative, graph_summary)
        l1 = self.loss(pos_graph, torch.ones_like(pos_graph))
        l2 = self.loss(neg_graph, torch.zeros_like(neg_graph))
        l = 1 * (l1+l2)

        # 引领树聚类
        c_lt, mu, r = lt_c(positive.squeeze(0), center_index, self.beta)
        cluster_summary = torch.sigmoid(r @ mu)
        if torch.cuda.is_available():
            cluster_summary = cluster_summary.cuda()
        pos_cluster = self.discriminatorLT(positive, cluster_summary)
        neg_cluster = self.discriminatorLT(negative, cluster_summary)
        
        l += self.alpha * (self.loss(pos_cluster, torch.ones_like(pos_cluster)) + self.loss(neg_cluster, torch.zeros_like(neg_cluster)))

        return l



def lt_c(data, center_index, beta):
    h1_c = data.cpu().detach().numpy()
    # h1, center_index, category = leadt(data, self.dc, self.K)
    # h1_c = h1

    h1_c = torch.tensor(h1_c)
    # center_index = center_index.copy()
    # center_index = torch.tensor(center_index)
    h1_c = h1_c / (h1_c.norm(dim=1)[:, None] + 1e-6)
    mu = h1_c[center_index]
    mu = mu / (mu.norm(dim=1)[:, None] + 1e-6)
    dist = torch.mm(h1_c, mu.transpose(0, 1))
    r = F.softmax(beta * dist, dim=1)

    # for i in range(h1.shape[0]):
    #     h1_c[i] = h1[category[i]]
    # h1_c = torch.tensor(h1_c)
    return h1_c, mu, r


def leadt(data, dc, k):   # data:经过gcn后的嵌入， k：
    tempM = np.sum(data ** 2, 1, dtype='float32').reshape(-1, 1)  ##The number of rows is unknown, only the number of columns is 1
    tempN = np.sum(data ** 2, 1, dtype='float32')  # X2 ** 2: element-wise square, sum(_,1): Adds in row direction, but ends up with row vectors
    sqdist = tempM + tempN - 2 * np.dot(data, data.T).astype('float32')
    sqdist[sqdist < 0] = 0
    D = np.sqrt(sqdist)
    lt1 = lt.LeadingTree(X_train=data, dc=dc, lt_num=k, D=D)
    lt1.fit()
    center_index = lt1.gamma_D[0:k]
    category = lt1.treeID

    return data, center_index, category

    
class Classifier(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret
