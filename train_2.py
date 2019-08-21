'''
Training the Image Quality Asessment network
Author: Ashish Gupta
Email: ashishagupta@gmail.com
'''

from setting import parse_opts 
from datasets.abide1 import Abide1Dataset
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time

from scipy import ndimage
import os

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    
    loss_class = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_class = loss_class.cuda()
    
    criterion = nn.CrossEntropyLoss()    
    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        print(f'epoch no: {epoch}')
        
        scheduler.step()
        
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, labels = batch_data
            print(f'batch: {batch_id}, volume shape: {volumes.shape}')
            if not sets.no_cuda: 
                volumes = volumes.cuda()

            optimizer.zero_grad()
            output = model(volumes)

            loss = criterion(output, labels)
            
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            
          
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimeizer': optimizer.state_dict()},
                                model_save_path)
                            
    print('Finished training')            
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
     
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    print (model)
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-2)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
        
    training_dataset = Abide1Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
