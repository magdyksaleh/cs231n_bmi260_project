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
    def __init__(self, csv_file, root_dir, transform=None, train=False):
        
        #args: csv_file path and filename of file
        #      root_Dir dir to dataset

        self.slice_labels = np.asarray(pd.read_csv(csv_file, header=None))
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
    
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
                    return slice_path, int(list_of_scans[scan_num])
                cntr += 1
         
    def __getitem__(self, idx):
        slice_path, scan_num = self.find_slice_path(idx)
        ds=pydicom.read_file(slice_path)
        hu_img = ds.RescaleIntercept + ds.pixel_array*ds.RescaleSlope
        # if(hu_img.shape != (512,512)):
        hu_img = transform.resize(hu_img, (64, 64), mode='constant')

        #grab label
        label = self.slice_labels[np.where(self.slice_labels[:,0] == scan_num)][0][1]
        
        sample = (np.asarray(hu_img), label)
        if self.transform:
            sample = self.transform(sample)
        return sample
