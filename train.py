"Implementation is based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi"

import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, Coauthor, AmazonCoBuy
from dgl import add_self_loop, remove_self_loop

from model import OLF_GCL, Classifier, leadt
import scipy.sparse as sp
from collections import Counter
import random
from sklearn.preprocessing import OneHotEncoder
from statistics import mean, stdev

from utils import load, encode_feat
from random_seed import setup_seed


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def main(args):

    setup_seed(35536)
    g, features, labels, n_classes, train_mask, val_mask, test_mask = load(args.dataset)
    in_feats = features.shape[1]

    g = remove_self_loop(g)
    g = add_self_loop(g)

    accs = []
    t_st = time.time()

    # 选点
    h = encode_feat(g, features, in_feats)
    h = h.cpu().detach().numpy()
    h_1, center_index, category = leadt(h, args.dc, args.tree_nums)
    center_index = center_index.copy()
    center_index = torch.tensor(center_index)

    if args.gpu < 0:
        cuda = False

    else:
        cuda = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        g = g.to(device)
        center_index = center_index.to(device)

    model = OLF_GCL(g, in_feats, args.n_hidden, args.n_layers, nn.PReLU(args.n_hidden), args.dropout, args.beta, args.alpha)

    if cuda:
        model.to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.olf_lr, weight_decay=args.wd1)

    cnt_wait = 0
    best = 1e9
    dur = []
    for epoch in range(args.n_model_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        model_optimizer.zero_grad()
        loss = model(features, center_index)
        #print(loss)
        loss.backward()
        model_optimizer.step()

        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_OLF_GCL.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            #print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch:", epoch, "loss:", loss.item())
        # logger.info(f"Epoch: {epoch}, loss: {loss.item()}")

    # train classifier
    #print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_OLF_GCL.pkl'))
    embeds = model.encoder(features, corrupt=False)
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    # embeds = embeds / (embeds+ 1e-8).norm(dim=1)[:, None]
    embeds = embeds.detach()

    dur = []
    classifier = Classifier(args.n_hidden, n_classes)
    x_bent = nn.CrossEntropyLoss()
    if cuda:
        classifier.to(device)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr, weight_decay=args.wd2)
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = x_bent(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_acc = evaluate(classifier, embeds, labels, val_mask)
        test_acc = evaluate(classifier, embeds, labels, test_mask)
        print('Epoch:', epoch, 'val_acc:', val_acc, 'test_acc', test_acc)

    accs.append(test_acc)
    print(test_acc)

    print('=================== time', int((time.time() - t_st)/60))
    return mean(accs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OLF-GCL')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--olf-lr", type=float, default=1e-3,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-model-epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--wd1", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--wd2", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=50,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument('--beta', dest='beta', type=int, default=10, help='')
    parser.add_argument('--alpha', dest='alpha', type=int, default=10, help='')
    parser.add_argument('--dc', dest='dc', type=float, default=0.06, help='')
    parser.add_argument('--tree_nums', type=int, default=16, help='')
    # parser.add_argument('--d', dest='dataset', type=str, default='cora', help='')


    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    a = []
    for i in range(2):
        acc = main(args)
        a.append(acc)

    print(a)
    print(args.dataset, ' Acc (mean)', mean(a), ' (std)', stdev(a))
