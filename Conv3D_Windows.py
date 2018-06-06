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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Create dataset class
class ILDDataset(Dataset):
    def __init__(self, csv_file, root_dir, mask=False,  transform=None, train=False, HU=True, resize=64):
        
        #args: csv_file path and filename of file
        #      root_Dir dir to dataset

        self.slice_labels = np.asarray(pd.read_csv(csv_file, header=None))
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.HU = HU
        self.batch_size = 4  
        self.resize = resize 
        if self.train:
            self.len = 1982 - self.batch_size + 1 #manually calculated
        else:
            self.len = 375 - self.batch_size +1  #manually calculated 
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
            try:
                list_of_slices.remove("_listing")
            except ValueError:
                pass
            
            try:
                list_of_slices.remove("_DS_Store")
            except ValueError:
                pass
            
            list_of_slices.sort()
            num_slices = len(list_of_slices)
            slice_cntr = 0
            num_rejected = 0
            for slice_name in list_of_slices:
                if (slice_name[-4:] != ".dcm"):
                    slice_cntr += 1
                    num_rejected += 1
                    continue
                #print(slice_name)
                slice_path = os.path.join(scan_path, slice_name)
                if (cntr >= idx) and  (slice_cntr < (num_slices - self.batch_size)) and (slice_name[-4:] == '.dcm'):
                    #print(slice_name[-4:])
                    return slice_path, int(list_of_scans[scan_num]), scan_path, slice_cntr, (slice_cntr - num_rejected + 1)
                slice_cntr += 1
                cntr += 1
    
    def find_mask_path(self, scan_path, slice_num):
        mask_path = os.path.join(scan_path,"lung_mask")
        for mask in os.listdir(mask_path):
            if (mask[-4:] != ".dcm"):
                continue
            if(mask[-6:-4].isdigit()):
                if(int(mask[-6:-4]) == slice_num):
                    return os.path.join(mask_path, mask)
            elif((slice_num<10) and (mask[-5:-4].isdigit()) and (int(mask[-5:-4]) == slice_num)):
                return os.path.join(mask_path, mask)
         
    def __getitem__(self, idx):
        filtered_imgs = np.zeros((4, self.resize, self.resize))
        slice_path, scan_num, scan_path, slice_num, mask_idx = self.find_slice_path(idx)
        list_of_slices = os.listdir(scan_path)
        
        try:
            list_of_slices.remove("_listing")
        except ValueError:
            pass
            
        try:
            list_of_slices.remove("_DS_Store")
        except ValueError:
            pass
        
        list_of_slices.sort()
        for i in range(self.batch_size):
            if(i != 0):
                slice_path = os.path.join(scan_path, list_of_slices[slice_num + i])
            #print(slice_path)
            mask_path = self.find_mask_path(scan_path, mask_idx+i)
            ds = pydicom.read_file(slice_path)
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
            filtered_imgs[i, :] = filtered_im

        #grab label
        label = self.slice_labels[np.where(self.slice_labels[:,0] == scan_num)][0][1]
        sample = (filtered_imgs, label)
        if self.transform:
            sample = self.transform(sample)
        return sample
