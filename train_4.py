'''
Training the Image Quality Asessment network
Author: Ashish Gupta
Email: ashishagupta@gmail.com
'''

from setting import parse_opts 
from model import generate_model
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from abide1dataset import ABIDE1

from torchviz import make_dot, make_dot_from_trace
import graphviz
from graphviz import Digraph, Source

import numpy as np
import os

def train(train_loader, model, optimizer, criterion):
    """train the ResNet model for IQA
    
    Arguments:
        epoch {int} -- iterator on set of training epochs
    """
    # set the model in training mode
    model.train()
    for epoch in range(1):
        print(f'len train_loader: {len(train_loader)}')
        for batch_idx, (data, target) in enumerate(train_loader):
            # data = data.unsqueeze(1)
            # print(torch.cuda.max_memory_allocated(device='cuda:0')/100000)
            # data = Variable(data)
            target = target.to('cuda:0')
            # target = Variable(target)
            # zero the gradients
            # optimizer.zero_grad()
            # output = model(data)
            
            dot = make_dot(model(data), params=dict(model.named_parameters()))
            dot.format = 'png'
            dot.render('model.png')

            break
            
            
            # loss = criterion(output, target)
            
            # # print(f'batch id: {batch_idx}, loss.data: {loss.item()}')
            # loss.backward()
            # for p in model.parameters():
            #     p.grad = None
            # optimizer.step()
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
            # del output, loss

        # save snap-shot : To Do


def test(test_loader, model, criterion):
    # set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    _n_test_images = len(test_loader.image_path_list)
    for data, target in test_loader():
        data = data.unsqueeze(1)
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= _n_test_images
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, _n_test_images,
        100. * correct / _n_test_images))





def main():

    # settting
    sets = parse_opts()
    model, parameters = generate_model(sets) 

    #debug
    # print(model)

    # print(f'type(model):{type(model)}')
    # print(f'type(parameters):{type(parameters)}')
    # print(f'type(model.parameters):{type(model.parameters)}')

    # freeze the pre-trained convolutional network parameters
    # for param in parameters['base_parameters']:
    #     param.requires_grad = False

    # ----------------------------------------------------
    # Debug:
    # for key in parameters.keys():
    #     _list = parameters[key]
    #     for i, item in enumerate(_list):
    #         print(item.shape)
    #         print(item.requires_grad)

    # criterion
    criterion = CrossEntropyLoss()
    # optimizer
    params = [{ 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }]

    optimizer = optim.Adam(params, lr=0.1,)

    # train loader
    train_dataset = ABIDE1(sets, train=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)

    # test loader
    # test_dataset = ABIDE1(sets, train=False)
    # test_loader = DataLoader(test_dataset, num_workers=1)


    
    train(train_loader, model, optimizer, criterion)
    # test(test_loader, model, criterion)
    

if __name__ == '__main__':
    main()

