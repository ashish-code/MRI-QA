"""
Multi-Layer Perceptron model for Image Quality Assessment
The data consists of Image Quality Metrics (IQM) for each subject in ABIDE 1 and DS030

Author: Ashish Gupta
Email: ashishagupta@gmail.com
"""

import torch 
import torch.nn as nn 
from torchvision import transforms
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler 
from torch.utils.data import Dataset
import torch.nn.functional as F 

import pandas as pd
import numpy as np 
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='train or validation')
    parser.add_argument('--data_file', default='x_abide.csv', help='data csv file')
    parser.add_argument('--train_file', default='train_2.csv', help='set of subjects for training')
    parser.add_argument('--val_file', default='val_2.csv', help='set of subjects for validation')
    parser.add_argument('--test_data_file', default='x_ds030.csv', help='test data csv file')
    parser.add_argument('--test_label_file', default='y_ds030.csv', help='test label csv file')
    args = parser.parse_args()
    return args


class AbideIQMDataSet(Dataset):
    """
    class image quality metrics on the abide 1
    inherits from torch Dataset
    """
    def __init__(self, options):
        self.phase = options.phase
        df_x = pd.read_csv(options.data_file)
        df_x = df_x.as_matrix()
        data_file = ''
        if options.phase == 'train':
            data_file = options.train_file
        else:
            data_file = options.val_file
        
        df_sub = pd.read_csv(data_file, header=None)
        df_sub = df_sub.as_matrix()
        
        X = np.empty((df_sub.shape[0], df_x.shape[1]-1), dtype=np.float32)
        
        y = df_sub[:,-1]
        for i in range(df_sub.shape[0]):
            idx = df_sub[i,0]
            _t = df_x[df_x[:,0]==idx]
            _t = _t.tolist()
            print(len(_t))
            X[i,:] = np.array(_t[1:])
        
        self.data = X
        self.label = y

    def __len__(self):
        n_row,_ = self.data.shape
        return n_row

    def __getitem__(self, index):
        return self.data[index,:], self.label[index]


class DSDataSet(Dataset):
    """
    Dataset class for DS030
    Inherits from Torch Dataset
    """
    def __init__(self, options):
        self.phase = options.phase
        assert(self.phase == 'test')
        df_x = pd.read_csv(options.test_data_file)
        df_x = df_x.as_matrix()
        df_y = pd.read_csv(options.test_label_file)
        y_sub_ids = df_y['subject_id'].tolist()
        y_labels = df_y['rater_1'].tolist()

        y_label = []
        y_sub = []
        for idx, y_lbl in enumerate(y_labels):
            if y_lbl == +1:
                y_label.append(1)
                y_sub.append(y_sub_ids[idx])
            elif y_lbl == -1:
                y_label.append(0)
                y_sub.append(y_sub_ids[idx])
        X = np.empty((len(y_sub), df_x.shape[1]-1), dtype=np.float32)
        for id, sub_id in enumerate(y_sub):
            X[id,:] = df_x[sub_id, 1:]
        
        self.data = X
        self.label = np.array(y_label)

    def __len__(self):
        n_row,_ = self.data.shape
        return n_row

    def __getitem__(self, index):
        return self.data[index,:], self.label[index]


class IQMMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 4*n_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4*n_features),
            nn.Dropout(0.25),
            nn.Linear(4*n_features, 2*n_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2*n_features),
            nn.Dropout(0.25),
            nn.Linear(2*n_features, n_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.25),
            nn.Linear(n_features, n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.model(x)





if __name__=='__main__':
    options = parse_opts()

    # dataset object
    abidedataset = AbideIQMDataSet(options)
    train_loader = DataLoader(abidedataset, batch_size=64, shuffle=True)

    for data, target in train_loader:
        print(data, target)
    # dataloaders

