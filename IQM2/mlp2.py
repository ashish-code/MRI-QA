import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNet, NeuralNetClassifier
from skorch.dataset import CVSplit

import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt 
import pickle

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='train/validation/test')
    parser.add_argument('--train_file', default='./abide.csv', help='training data and labels')
    parser.add_argument('--test_file', default='./ds030.csv', help='test data and labels')
    parser.add_argument('--n_epochs', default=20000, help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate for the optimizer')
    parser.add_argument('--chkpt_dir', default='./checkpoints/', help='checkpoint directory')
    args = parser.parse_args()
    return args

class IQMMLP(nn.Module):
    def __init__(self, n_features=65, n_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.25),

            nn.Linear(50, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.25),            

            # nn.Linear(30, 20),
            # nn.BatchNorm1d(20),
            # nn.ReLU(inplace=True),


            # nn.Linear(20, 5),
            # nn.BatchNorm1d(5),
            # nn.ReLU(inplace=True),

            
            nn.Linear(20, n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    options = parse_opts()

    raw_data = pd.read_csv(options.train_file, header=None)
    raw_data = raw_data.as_matrix()
    X = raw_data[:,:-1]
    X = X.astype(np.float32)
    y = raw_data[:,-1]
    y = y.astype(np.int64)

    # print(X.shape)
    # print(y.shape)

    net = NeuralNetClassifier(IQMMLP, 
    max_epochs = options.n_epochs, 
    lr=options.learning_rate,
    iterator_train__shuffle=True,
    optimizer=torch.optim.Adam,
    batch_size=32,
    iterator_valid__shuffle=True,
    iterator_train__num_workers=2,
    iterator_valid__num_workers=2,
    device='cuda:0',
    criterion=nn.CrossEntropyLoss,
    train_split=CVSplit(5, stratified=True)
    )

    net.fit(X,y)

    # save the model
    net.save_params(f_params='./checkpoints/abide_3.pkl')

