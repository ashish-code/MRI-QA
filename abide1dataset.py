"""Use PyTorch's Dataset class to create a custom class for ABIDE 1 dataset
We crop random sample volumes from each subject as part of dataloader

Author: Ashish Gupta
Email: ashishagupta@gmail.com
"""

import os
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import nibabel


class ABIDE1(Dataset):
    def __init__(self, settings, train=True):
        """Create a custom dataset object
        
        Arguments:
            Dataset {torch.utils.data.dataset.Dataset} -- Dataset handing object
            settings {Setting} -- optional arguments regarding the data
        
        Keyword Arguments:
            data_list_file_path {str} -- [train/val list of paths and labels] (default: {'data/ABIDE1/train.csv'})
            data_root_dir {str} -- [URL of the data] (default: {'/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/'})
        """
        self.phase = train                  # train/validation phase

        if self.phase:
            self.data_list_file_path = settings.train_list
            self.input_D = settings.input_D     # Depth
            self.input_H = settings.input_H     # Height
            self.input_W = settings.input_W     # Width
        else:
            self.data_list_file_path = settings.test_list
            self.input_D = 2*settings.input_D     # Depth
            self.input_H = 2*settings.input_H     # Height
            self.input_W = 2*settings.input_W     # Width

        
        path_list = []
        label_list = []

        with open(self.data_list_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            path, label = line.split(',')
            path_list.append(str(path))
            label_list.append(int(label))

        self.image_path_list = path_list
        self.image_label_list = label_list

    def __getitem__(self, index):
        """returns tuple of image pytorch tensor and associated label
        
        Arguments:
            index {int} -- index of item to be retrieved from dataset
        """
        nii = nibabel.load(self.image_path_list[index])
        label = self.image_label_list[index]

        #debug:
        # print(f'1: nii_shape: {nii.shape}')
        # print(f'1: nii type: {type(nii)}')
        
        nii = nii.get_fdata()

        # print(f'2: nii_shape: {nii.shape}')
        # print(f'2: nii type: {type(nii)}')

        # random sample a [H, W, D] volume
        nii = self.__random_crop__(nii)
        # convert to tensor
        nii_tensor = torch.tensor(nii)
        # add leading dimension for batch
        # nii_tensor = nii_tensor.unsqueeze(0)

        return nii_tensor, label

    
    def __random_crop__(self, data):
        """ Crop a volume based on the desired height, width, and depth specified in settings
        
        Arguments:
            data {np.array} -- [crop array to input H,W,D]
        """
        [img_h, img_w, img_d] = data.shape
        # The MRI contains extraneous slices at the top and bottom in the Axial perspective
        # we are intentionally cropping out the top and bottom of the Axial slices
        top_D_to_remove = 0
        bottom_D_to_remove = 0
        [max_H, max_W, max_D] = [img_h, img_w, img_d-top_D_to_remove]
        [min_H, min_W, min_D] = [0, 0, bottom_D_to_remove]
        
        in_D = self.input_D
        in_H = self.input_H
        in_W = self.input_W

        D_0 = int(random.randint(min_D+1, max_D-in_D-1))
        H_0 = int(random.randint(min_H+1, max_H-in_H-1))
        W_0 = int(random.randint(min_W+1, max_W-in_W-1))

        D_1 = int(D_0 + in_D)
        H_1 = int(H_0 + in_H)
        W_1 = int(W_0 + in_W)

        return data[H_0:H_1, W_0:W_1, D_0:D_1]

    
    def __len__(self):
        return len(self.image_label_list)


    def __repr__(self):
        """Overloading the representation string for debuggin in code development phase
        
        Returns:
            [str] -- class representation
        """
        _str = '\n'.join([item for item in self.image_path_list])
        return _str


# --------------------------------------------------
# DEBUG
# --------------------------------------------------


class Setting():
    def __init__(self):
        self.input_D = 64
        self.input_H = 64
        self.input_W = 64
        self.phase = 'train'

def test_abide1(setting):
    abide1 = ABIDE1(setting)
    print(abide1)


def test_io_random(setting):
    abide1 = ABIDE1(setting)
    rnd_idx = random.randint(0, abide1.__len__())
    print(f'random index: {rnd_idx}')
    img, lbl = abide1.__getitem__(rnd_idx)
    print(f'shape: {img.shape}, label: {lbl}')


def test_dataloader(setting):
    abide1 = ABIDE1(setting)
    data_loader = DataLoader(abide1, batch_size=8, shuffle=True)
    # images, labels = iter(data_loader).next()
    # print(f'images shape: {images.shape}, labels shape: {labels.shape}')
    print(f'number of batches: {len(data_loader)}')

    # for batch_id, batch_data in enumerate(data_loader):
    #     images, labels = batch_data
    #     print(f'batch id: {batch_id}, images shape: {images.shape}, labels shape: {labels.shape}')



if __name__ == '__main__':
    # test_abide1(Setting())
    # test_io_random(Setting())
    test_dataloader(Setting())



