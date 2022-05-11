import torch.nn as nn
import torch.nn.functional as F
from tool_GAAT.layer_GAAT import GCNLayer, GATLayer
import torch
from numpy import math
from tool_GAAT import utils, adj
import numpy as np
from torch.nn.parameter import Parameter


class GAAT(nn.Module):
    """
    GCN 62*5 --> 62*(2+4+6+8)
    """
    def __init__(self, in_feature, augment_feature, nclass, dropout, lrelu, alpha, adj_matrix):
        super(GAAT, self).__init__()

        self.in_feature = in_feature
        self.augment_feature = augment_feature
        # self.out_feature = out_feature
        self.nclass = nclass
        self.dropout = dropout
        self.l_relu = lrelu
        self.alpha = alpha
        self.adj = adj_matrix

        # graph attention layer
        self.att = GATLayer(self.in_feature, self.augment_feature, self.l_relu, self.dropout)

        # GCN layer
        self.gcn_1 = GCNLayer(self.in_feature, 10, False) # 5->20
        self.gcn_2 = GCNLayer(10, 15, False) # 8->16

        self.mlp = nn.Linear(62*30, 128)
        self.mlp2 = nn.Linear(128, nclass)

        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        laplacian = adj.normalize_adj(self.adj)

        delta_lap = self.lrelu(self.att(x))# attention update laplacian
        zero_vec = -9e15*torch.ones_like(delta_lap)
        attention = torch.where(laplacian.data > 0.1, delta_lap, zero_vec)
        attention = F.softmax(attention, dim=1)

        laplacian = laplacian*(1-self.alpha) + attention*self.alpha

        x1 = self.lrelu(self.gcn_1(x, laplacian))
        x2 = self.lrelu(self.gcn_2(x1, laplacian))
        x = torch.cat((x, x1, x2), 2)

        x = x.view(x.size(0),-1)

        x = self.lrelu(self.mlp(x))
        x = self.bn(x)

        x = self.mlp2(x)
        
        return x, laplacian










