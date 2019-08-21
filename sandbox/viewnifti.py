import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

nii_path = '../data/MRBrainS18/1/reg_T1.nii.gz'

img = nibabel.load(nii_path)
print(img.shape)



abide_nii_path = '../sub-0050551_T1w.nii.gz'
img_abide = nibabel.load(abide_nii_path)
print(img_abide.shape)