from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pydicom

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Create dataset class
class ILDDataset(Dataset):
    def __init__(self, csv_file, root_dir, mask_dir, mask=False,  transform=None, train=False):
        
        #args: csv_file path and filename of file
        #      root_Dir dir to dataset

        self.slice_labels = np.asarray(pd.read_csv(csv_file, header=None))
        self.root_dir = root_dir
        self.mask_label_dir = mask_dir
        self.transform = transform
        self.train = train
        self.mask = mask
    
    def __len__(self):
        return len(self.slice_labels)

    def find_slice_path(self, idx):
        list_of_scans = os.listdir(self.root_dir)
        num_scans = len(list_of_scans)
        cntr = 0
        for scan_num in range(num_scans):
            scan_path = os.path.join(self.root_dir,list_of_scans[scan_num])
            if (not os.path.isdir(scan_path)) or (list_of_scans[scan_num] == "HRCT_pilot"):
                continue
            list_of_slices = os.listdir(scan_path)
            num_slices = len(list_of_slices)
            for slice_num in range(num_slices):
                if (list_of_slices[slice_num][-4:] != ".dcm"):
                    continue
                slice_path = os.path.join(scan_path,list_of_slices[slice_num])
                if (cntr == idx):
                    return slice_path, int(list_of_scans[scan_num]), slice_num, scan_path, list_of_slices[slice_num]
                cntr += 1
    
    def find_mask_path(self, scan_path, slice_name):
        mask_path = os.path.join(scan_path,"lung_mask")
        slice_num = int(slice_name[-6:-4])
        for mask in os.listdir(mask_path):
            if (mask[-4:] != ".dcm"):
                continue
            if(mask[-6:-4].isdigit()):
                if(int(mask[-6:-4]) == slice_num):
                    return os.path.join(mask_path, mask)
            elif((slice_num<10) and (mask[-5:-4].isdigit()) and (int(mask[-5:-4]) == slice_num)):
                return os.path.join(mask_path, mask)
         
    def __getitem__(self, idx):
        slice_path, scan_num, slice_num, scan_path, slice_name = self.find_slice_path(idx)
        mask_path = self.find_mask_path(scan_path, slice_name)
        ds=pydicom.read_file(slice_path)
        hu_img = ds.RescaleIntercept + ds.pixel_array*ds.RescaleSlope
        if ((mask_path is not None ) and (self.mask == True)):
            mask=pydicom.read_file(mask_path).pixel_array
        else:
            mask=np.ones_like(hu_img)
        filtered_im = np.asarray(hu_img)*np.asarray(mask)
        filtered_im = transform.resize(filtered_im, (64, 64), mode='constant')

        #grab label
        label = self.slice_labels[np.where(self.slice_labels[:,0] == scan_num)][0][1]
        sample = (filtered_im, label)
        if self.transform:
            sample = self.transform(sample)
        return sample
