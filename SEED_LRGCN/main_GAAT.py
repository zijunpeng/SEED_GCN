from __future__ import division
from __future__ import print_function
import math

import os
import time
import h5py
import argparse

import numpy as np
from scipy import sparse as sp
import torch
# from chord import Chord
import pandas as pd
import plotly.express as px
import pylab

from torch import float32, float64
from torch.autograd import Variable
from torch.utils.data import dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter

from torch.utils.tensorboard import SummaryWriter
# write_path = SummaryWriter('/home/ming/workspace/test_GAAT/GAAT/tensorboard')

from tool_GAAT import adj, model_GAAT
from tool_GAAT.early_stopping import EarlyStopping
from tool_GAAT import pre_process
# from torchsummary import summary


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--augment_feature_attention', type=int, default=20,
                    help='Feature augmentation matrix for attention.')
parser.add_argument('--out_feature', type=int, default=20, help='Output feature for GCN.')
parser.add_argument('--seed', type=int, default=56, help='Random seed.') #42
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate .')
parser.add_argument('--leakyrelu', type=float, default=0.15, help='leaky relu.')
parser.add_argument('--k_i', type=int, default=3, help='k_i order chebyshev.')
parser.add_argument('--alpha', type=float, default=0.2, help='Attention reconciliation hyperparameters') #5e-4
parser.add_argument('--beta', type=float, default=5e-5, help='update laplacian matrix') #5e-4
parser.add_argument('--patience', type=int, default=20, help='early stopping param')
parser.add_argument('--model_save_path', type=str, default='/home/ming/workspace/LRGCN/GAAT/modelSave/', help='')
parser.add_argument('--data_path', type=str, default='/home/EGG_Data/SEED/ExtractedFeatures/1/', help='')
parser.add_argument('--random_adj_matrix', type=bool, default=False, help='')


args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device is", device)


# summary(model, input_size=(62, 5))
def train_model(subject, data_path):
    print("current subject is " + subject)

    #####################################################################################
    #1.load data
    #####################################################################################
    X_train, X_test, y_train, y_test = pre_process.data_intra_sub_single_fre((data_path+subject), "de_LDS")
    
    train_set = TensorDataset((torch.from_numpy(X_train)).type(torch.FloatTensor), (torch.from_numpy(y_train)).type(torch.FloatTensor))
    val_set  =  TensorDataset((torch.from_numpy(X_test)).type(torch.FloatTensor), (torch.from_numpy(y_test)).type(torch.FloatTensor))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False)

    adj_matrix = Parameter(torch.FloatTensor(adj.draw_adj()))

    #####################################################################################
    #2.define model
    #####################################################################################
    model = model_GAAT.GAAT(
            in_feature=X_train.shape[2], # 5
            augment_feature = args.augment_feature_attention, # attention augmentation feature
            nclass=3,
            dropout=args.dropout,
            lrelu = args.leakyrelu,
            alpha = args.alpha,
            adj_matrix = adj_matrix,
            )

    weight_params = []
    lap_params = []
    for pname, p in model.named_parameters():
        if str(pname) == "adj":
            lap_params += [p]
        else:
            weight_params += [p]


    optimizer = optim.Adam(weight_params,
                        lr=args.lr, weight_decay=args.weight_decay)
    _loss = nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    earlyStopping = EarlyStopping(args.patience, verbose=True, path=args.model_save_path)

    #############################################################################
    #3.start train
    #############################################################################
    t_total = time.time()
    train_epoch = args.epochs
    best_val_acc = 0

    for epoch in range(train_epoch):
        epoch_start_time = time.time()
        train_acc = 0
        train_loss = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for i,(x,y) in enumerate(train_loader):
            model.zero_grad()
            x,y = x.to(device), y.to(device=device, dtype=torch.int64)
            batch_x, batch_y = Variable(x), Variable(y)
            output, lap_1 = model(batch_x)
            loss = _loss(output, batch_y)
            loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
            train_loss += loss.item()

        model.eval()
        val_start_time = time.time()
        
        with torch.no_grad():
            for j,(a,b) in enumerate(val_loader):
                a,b = a.to(device), b.to(device=device, dtype=torch.int64)
                val_x, val_y = Variable(a), Variable(b)
                output, lap = model(val_x) 
                val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == val_y.cpu().data.numpy())
                batch_loss = _loss(output, val_y)
                val_loss += batch_loss.item()

        val_acc = val_acc / val_set.__len__()
        earlyStopping(subject, val_acc, model, epoch)

        if best_val_acc < val_acc:
            best_val_acc = val_acc

        if best_val_acc == 1:
            break

    return best_val_acc


acc_list = []
acc_dic = {}
for subject in os.listdir(args.data_path): 
    subjectName = str(subject)

    valAcc = train_model(subjectName, args.data_path)
    acc_list.append(valAcc)
    acc_dic[subjectName] = valAcc
    print(str(subjectName) + ":" + str(valAcc))


acc_list = np.array(acc_list)
print(acc_dic)
print("Average is : "+str(np.mean(acc_list)))
print("Standard deviation is : "+str(np.std(acc_list, ddof = 1)))






