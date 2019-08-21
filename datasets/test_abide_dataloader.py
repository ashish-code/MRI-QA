import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 

from abide1 import Abide1Dataset

data_root = '/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/'
img_list = '../data/ABIDE1/train.csv'

class MySetting:
    def __init__(self,):
        self.input_D = 64
        self.input_H = 64
        self.input_W = 64
        self.phase = 'train'
        self.data_root = '/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/'
        self.img_list = '../data/ABIDE1/train.csv'
        self.batch_size = 2
        self.num_workers = 2
        self.pin_memory = False

sets = MySetting()
training_dataset = Abide1Dataset(sets.data_root, sets.img_list, sets)
data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

# for i in range(10):
#     img, label = training_dataset.__getitem__(i)
#     print(img.shape)

itr = iter(data_loader)

item = next(itr)
type(item)