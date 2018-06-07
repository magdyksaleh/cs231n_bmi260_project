from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pydicom
from scipy.ndimage import imread

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Create dataset class
class ILDDataset(Dataset):
    def __init__(self, cystic_path, root_dir, mask=False,  transform=None, train=False, HU=True, resize=64, verbose=False):
        
        #args: csv_file path and filename of file
        #      root_Dir dir to dataset

        self.cystic_path = cystic_path
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.HU = HU
        self.verbose = verbose
        self.resize = resize 
        if self.train:
            self.len = 830 #manually calculated
        else:
            self.len = 151 #manually calculated 
        self.mask = mask
    
    def __len__(self):
        return self.len

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

    def find_cystic_mask_path(self, scan_num, slice_name):
        mask_path = os.path.join(self.cystic_path, str(scan_num))
        slice_num = int(slice_name[-6:-4])
        for mask in os.listdir(mask_path):
            if (mask[-4:] != ".png"):
                continue
            if(mask[-6:-4].isdigit()):
                if(int(mask[-6:-4]) == slice_num):
                    return os.path.join(mask_path, mask)
            elif((slice_num<10) and (mask[-5:-4].isdigit()) and (int(mask[-5:-4]) == slice_num)):
                return os.path.join(mask_path, mask)
         
    def __getitem__(self, idx):
#         print(idx)
        slice_path, scan_num, slice_num, scan_path, slice_name = self.find_slice_path(idx)
        if(self.verbose):
            print(slice_path)
        mask_path = self.find_mask_path(scan_path, slice_name)
        cyst_mask_path = self.find_cystic_mask_path(scan_num, slice_name)
        ds=pydicom.read_file(slice_path)
        if self.HU:
            hu_img = ds.RescaleIntercept + ds.pixel_array*ds.RescaleSlope
        else:
            hu_img = ds.pixel_array
        if ((mask_path is not None )and (self.mask == True)):
            mask=pydicom.read_file(mask_path).pixel_array
            mask[mask>0] = 1
        else:
            mask=np.ones_like(hu_img)
        filtered_im = np.asarray(hu_img)*np.asarray(mask)
        filtered_im = transform.resize(filtered_im, (self.resize, self.resize), mode='constant')

        #grab label
        if(cyst_mask_path is None):
            label = np.zeros_like(filtered_im)
        else:
            label = np.asarray(imread(cyst_mask_path))
            if(len(label.shape) != 2):
                label = np.sum(label, axis=2)
                label = (label - np.mean(label))/np.std(label)
                label = label > 0
                
        label = transform.resize(label, (self.resize, self.resize), mode='constant')
#         print(slice_path)
        # print(label.shape)
        # print(filtered_im.shape)
        sample = (filtered_im, label)
        if self.transform:
            sample = self.transform(sample)
        return sample
