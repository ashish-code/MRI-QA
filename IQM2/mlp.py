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

for skorch import NeuralNetClassifier

import pandas as pd
import numpy as np 
import argparse
import os
import matplotlib.pyplot as plt 

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='train/validation/test')
    parser.add_argument('--train_file', default='./abide_train.csv', help='training data and labels')
    parser.add_argument('--val_file', default='./abide_val.csv', help='validation data and labels')
    parser.add_argument('--test_file', default='./ds030_test.csv', help='test data and labels')
    parser.add_argument('--n_epochs', default=40000, help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate for the optimizer')
    parser.add_argument('--chkpt_dir', default='./checkpoints/', help='checkpoint directory')
    args = parser.parse_args()
    return args


def balance_classes(df):
    """over-sampling for class balance
    
    Arguments:
        df {pd.DataFrame} -- header less pandas data frame
    
    Returns:
        pd.DataFrame -- dataframe with classes balanced 
    """
    df_class_0 = df[df[65]==0]
    df_class_1 = df[df[65]==1]
    df_count = df[65].value_counts()
    count_0 = df_count[0]
    count_1 = df_count[1]

    if count_0 > count_1:
        df_class_1_over = df_class_1.sample(count_0, replace=True)
        df_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    elif count_0 < count_1:
        df_class_0_over = df_class_0.sample(count_1, replace=True)
        df_over = pd.concat([df_class_1, df_class_0_over], axis=0)
    else:
        df_over = df
    
    return df_over


class IQMDataSet(Dataset):
    """
    class image quality metrics on the abide 1
    inherits from torch Dataset
    """
    def __init__(self, options, phase='train'):
        super().__init__()
        if phase=='train':
            # raw_data = np.genfromtxt(options.train_file, delimiter=',')
            raw_data = pd.read_csv(options.train_file, header=None)
        elif phase=='val':
            # raw_data = np.genfromtxt(options.val_file, delimiter=',')
            raw_data = pd.read_csv(options.val_file, header=None)
        elif phase =='test':
            # raw_data = np.genfromtxt(options.test_file, delimiter=',')
            raw_data = pd.read_csv(options.test_file, header=None)
        else:
            print(f'Phase provided {phase} not recognized. Enter: train/val/test.')
            exit()
        # balance the class labels in the data
        raw_data = balance_classes(raw_data)
        raw_data = raw_data.as_matrix()
        self.data = raw_data[:,:-1]
        self.label = raw_data[:,-1]

    def __len__(self):
        nrow, _ = self.data.shape
        return nrow

    def __getitem__(self, index):
        _data = self.data[index,:]
        _label = self.label[index]
        return _data, _label


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
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.bias)
        
    def forward(self, x):
        return self.model(x)


class IQMMLP2(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Linear(n_features, int(n_features/2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(n_features/2)),
            # nn.Dropout(0.1),
            nn.Linear(int(n_features/2), int(n_features/4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(n_features/4)),
            # nn.Dropout(0.1),
            nn.Linear(int(n_features/4), int(n_features/8)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(n_features/8)),
            nn.Linear(int(n_features/8), int(n_features/16)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(n_features/16)),
            # nn.Dropout(0.1),
            nn.Linear(int(n_features/16), n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        
    def forward(self, x):
        return self.model(x)


class IQMMLP3(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(50, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),            

            nn.Linear(30, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(20, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            # nn.Linear(10, 5),
            # # nn.BatchNorm1d(5),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(0.1),
            
            nn.Linear(10, n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        
    def forward(self, x):
        return self.model(x)


class IQMMLP4(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),            

            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            # nn.Linear(10, 5),
            # # nn.BatchNorm1d(5),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(0.1),
            
            nn.Linear(10, n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        
    def forward(self, x):
        return self.model(x)


def run_evaluation(model, val_loader):
    model.eval()
    # ---------------  Validation -------------------
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for raw_data in val_loader:
            data, label = raw_data
            data = data.float()
            label = label.long()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    val_accuracy = float(correct)/total
    return val_accuracy


def run_test(model, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for raw_data in test_loader:
            data, label = raw_data
            data = data.float()
            label = label.long()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    test_accuracy = float(correct)/total
    return test_accuracy
    # print('Accuracy on DS030 test: %f %%'%(100*correct/total))


def run_train(model, train_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for raw_data in train_loader:
            data, label = raw_data
            data = data.float()
            label = label.long()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    train_accuracy = float(correct)/total
    return train_accuracy



if __name__=='__main__':
    options = parse_opts()

    # dataset objects
    dataset_train = IQMDataSet(options, phase='train')
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_val = IQMDataSet(options, phase='val')
    n_validation = len(dataset_val)
    val_loader = DataLoader(dataset_val, batch_size=n_validation, shuffle=True, drop_last=True)

    dataset_test = IQMDataSet(options, phase='test')
    n_test = len(dataset_test)
    test_loader = DataLoader(dataset_test, batch_size=n_test, shuffle=True, drop_last=True)

    # model
    model = IQMMLP4(65, 2)
    model = model.float()

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    # put model in train mode

    # checkpoint file
    chkpt_file = options.chkpt_dir+'abide.pth'

    # if os.path.exists(chkpt_file):
    #     checkpoint = torch.load(chkpt_file)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    # else:
    #     model.train()
    #     epoch = 0
    #     while epoch < options.n_epochs:
    #         epoch += 1
    #         running_loss = 0.0
    #         batch_count = 0
    #         for batch_itr, (batch_data, batch_target) in enumerate(train_loader):
    #             optimizer.zero_grad()
    #             batch_count += 1

    #             outputs = model(batch_data.float())
    #             loss = criterion(outputs, batch_target.long())
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()

    #         if epoch%100 == 0:    
    #             print('epoch: %d, loss: %.6f' %(epoch, running_loss / batch_count))
        

    
    epoch = 0
    loss_trend = []
    train_loss_trend = []
    train_acc_trend = []
    val_acc_trend = []
    test_acc_trend = []

    while epoch < options.n_epochs:
        model.train()
        epoch += 1
        running_loss = 0.0
        batch_count = 0
        for batch_itr, (batch_data, batch_target) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_count += 1

            outputs = model(batch_data.float())
            loss = criterion(outputs, batch_target.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        loss_trend.append(running_loss)

        if epoch%100 == 0:
            train_loss = running_loss / batch_count
            train_accuracy = run_train(model, train_loader)
            val_accuracy = run_evaluation(model, val_loader)
            test_accuracy = run_test(model, test_loader)
            print('epoch: %d, loss: %.6f, train_acc: %.2f, val_acc: %.2f, test_acc: %.2f' %(epoch, train_loss, train_accuracy, val_accuracy, test_accuracy))

            train_loss_trend.append(train_loss)
            train_acc_trend.append(train_accuracy)
            val_acc_trend.append(val_accuracy)
            test_acc_trend.append(test_accuracy)
    
    
    print('completed training')
       
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, chkpt_file)

    # put model in evaluation mode
    # model.eval()
    # ---------------  Validation -------------------
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for raw_data in val_loader:
    #         data, label = raw_data
    #         data = data.float()
    #         label = label.long()
    #         output = model(data)
    #         _, predicted = torch.max(output.data, 1)
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()
    
    # print('Accuracy on ABIDE validation: %f %%'%(100*correct/total))


    # ----------------- Test -----------------
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for raw_data in test_loader:
    #         data, label = raw_data
    #         data = data.float()
    #         label = label.long()
    #         output = model(data)
    #         _, predicted = torch.max(output.data, 1)
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()
    
    # print('Accuracy on DS030 test: %f %%'%(100*correct/total))

    fig = plt.figure(1)
    x = list(range(1, len(train_loss_trend)+1))
    plt.plot(x, train_loss_trend, 'k-', label='x-entropy loss')
    plt.plot(x, train_acc_trend, 'b-', label='train acc')
    plt.plot(x, val_acc_trend, 'g-', label='validation acc')
    plt.plot(x, test_acc_trend, 'r-', label='DS030 test acc')
    plt.legend(loc='best')
    plt.title('DNN Training on ABIDE 1')
    plt.xlabel('epoch')
    plt.ylabel('perf measure')
    plt.savefig('loss_acc_trend.png')
    plt.show()



