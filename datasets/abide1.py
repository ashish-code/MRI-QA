'''
Dataset ABIDE 1

Author: ashish gupta
Email: ashishagupta@gmail.com
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

class Abide1Dataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        """Create dataset object
        
        Arguments:
            root_dir {[type]} -- [description]
            img_list {[type]} -- [description]
            sets {[type]} -- [description]
        """
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        # debug: echo number of subjects in train/val list
        print(f"Processing {len(self.img_list)} subjects")
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        """Number of subjects
        
        Returns:
            [type] -- [description]
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(',')
            img_name = ith_info[0]
            label = ith_info[1]
            # assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            # assert img is not None
            
            # data processing
            img_array = self.__training_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array, label
        
        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(",")
            img_name = ith_info[0]
            label = ith_info[1]
            # print(img_name)
            # assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            # assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array, label
            

    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area
        """
        # The values of min_z and max_z are specific abide 1, these hard coded values need to change!
        min_z = 75
        max_z = 225

        return volume[min_z:max_z, :, :]


    def __random_center_crop__(self, data):
        from random import random
        """
        Random crop
        """
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = [img_d, img_h, img_w]
        [min_D, min_H, min_W] = [0, 0, 0]
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())
        
        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))
       
        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])
 
        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)
        
        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data = self.__random_center_crop__ (data)
        
        return data

    def __training_data_process__(self, data): 
        # crop data according net input size
        data, label = data.get_data()
        
        # drop out the invalid range
        data = self.__drop_invalid_range__(data)
        
        # crop data
        data = self.__crop_data__(data) 

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label


    def __testing_data_process__(self, data): 
        # crop data according net input size
        data, label = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label
