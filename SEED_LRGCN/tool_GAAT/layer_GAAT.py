import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, input, lap):
        output = torch.matmul(lap, torch.matmul(input, self.weight))
        return output # (batch_size, 62, out_features)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GATLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, augment_feature, lrelu, dropout):
        super(GATLayer, self).__init__()

        self.in_features = in_features # input feature : 5
        self.out_features = augment_feature # output feature : 
        self.l_relu = nn.LeakyReLU(lrelu)
        self.relu = nn.ReLU()
        self.dropout = dropout

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features))) 
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1))) # 32 * 1
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h):
        # Wh.shape: (batch_size, N, out_features)
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features)
        a_input = self.do_attention(Wh)
        attention = self.l_relu(torch.matmul(a_input, self.a).squeeze(3))
        return attention


    def do_attention(self, Wh):
        # Wh.shape: (batch_size, N, out_features)
        data = []
        batch_size = Wh.size()[0]
        N = Wh.size()[1]

        for i in range(batch_size):
            sub_wh = Wh[i] # (N, out_features)
            Wh_repeated_in_chunks = sub_wh.repeat_interleave(N, dim=0)
            Wh_repeated_alternating = sub_wh.repeat(N, 1)
            # all_combinations_matrix.shape == (N, N, 2 * out_features)
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
            
            sub_data = all_combinations_matrix.view(1, N, N, 2 * self.out_features)
            data.append(sub_data)
        data = torch.cat(tuple(data), 0) # (batch_size, 62, 62, 2*out_feature)
        return data


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





