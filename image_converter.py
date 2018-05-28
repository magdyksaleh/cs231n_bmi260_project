##Convert images from dicom to png for labelling software

import numpy as np
import os
import pydicom
import png
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.misc import imsave

def convert_dir_to_png(root_dir, target_dir):
    list_of_scans = os.listdir(root_dir)
    num_scans = len(list_of_scans)
    for scan_num in tqdm(range(num_scans)):
        scan_path = os.path.join(root_dir,list_of_scans[scan_num])
        if (not os.path.isdir(scan_path)) or (list_of_scans[scan_num] == "HRCT_pilot"):
            continue
        list_of_slices = os.listdir(scan_path)
        num_slices = len(list_of_slices)
        for slice_num in range(num_slices):
            if (list_of_slices[slice_num][-4:] != ".dcm"):
                continue
            slice_path = os.path.join(scan_path,list_of_slices[slice_num])
            ds=pydicom.read_file(slice_path)
            hu_img  = ds.pixel_array
            # hu_img = ds.RescaleIntercept + ds.pixel_array*ds.RescaleSlope
            hu_img_np = np.asarray(hu_img)

            #need to navigate to correct directory 
            os.chdir(target_dir)
            new_scan_path = os.path.join(target_dir,list_of_scans[scan_num])
            if(not os.path.isdir(new_scan_path)):
                os.mkdir(new_scan_path)
            new_slice_path = os.path.join(new_scan_path, list_of_slices[slice_num])
            new_slice_name = new_slice_path[:-4]+".png"
            pngfile = open(new_slice_name, 'wb') 
            imsave(pngfile, hu_img_np)
            pngfile.close()




root_train_dir = "/Users/magdy/Desktop/BMI260/Project/Data/Cystic Dataset/Train"
target_train_dir = "/Users/magdy/Desktop/BMI260/Project/Data/Cystic Dataset_png/Train"
target_test_dir = "/Users/magdy/Desktop/BMI260/Project/Data/Cystic Dataset_png/Test"
root_test_dir = "/Users/magdy/Desktop/BMI260/Project/Data/Cystic Dataset/Test"

root_dir = root_train_dir
target_dir = target_train_dir

print("Converting Training Set")
convert_dir_to_png(root_dir, target_dir)

root_dir = root_test_dir
target_dir = target_test_dir

print("Converting Test Set")
convert_dir_to_png(root_dir, target_dir)