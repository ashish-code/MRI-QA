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

import numpy as np
import os

def train(epoch):
    """train the ResNet model for IQA
    
    Arguments:
        epoch {int} -- iterator on set of training epochs
    """
    # set the model in training mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        # zero the gradients
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        # save snap-shot : To Do


def test():
    # set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    _n_test_images = len(test_loader.image_path_list)
    for data, target in test_loader():
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


# settting
sets = parse_opts()
model, parameters = generate_model(sets) 

#debug
print(model)

# freeze the pre-trained convolutional network parameters
for param in parameters['base_parameters']:
    param.requires_grad = False

# ----------------------------------------------------
# Debug:
for key in parameters.keys():
    _list = parameters[key]
    for i, item in enumerate(_list):
        print(item.shape)
        print(item.requires_grad)

# ----------------------------------------------------
# + model
# + parameters
# + settings (configuration options)
# + dataloader (training)

# - optimizer
# - loss
# - checkpoint

# ---------------------------------------------------

# criterion
criterion = CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(parameters, lr=0.01)

# train loader
train_loader = ABIDE1(sets, train=True)

# test loader
test_loader = ABIDE1(sets, train=False)

for epoch in range(1,10):
    train(epoch)
    test()

