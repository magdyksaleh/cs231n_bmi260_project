##Convert images from dicom to png for labelling software

import numpy as np
import os
import pydicom
import png
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt
import skimage
from skimage import feature
from scipy.ndimage.morphology import binary_closing
from scipy.misc import imsave

def find_mask_path(scan_path, slice_name):
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

def generate_cyst_mask(root_dir, target_dir, thresh_hu):
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
            
            slice_name = list_of_slices[slice_num]
            slice_path = os.path.join(scan_path, slice_name)
            mask_path = find_mask_path(scan_path, slice_name)
            
            mask=np.asarray(pydicom.read_file(mask_path).pixel_array)
            ds = pydicom.read_file(slice_path)
            img=ds.pixel_array
            mask[mask>0] = 1
            img_filtered = img*mask
            intercept = ds.RescaleIntercept
            slope = ds.RescaleSlope
            img_filtered = img_filtered*slope + intercept
            img_filtered = np.float32(img_filtered)
            if(np.max(img_filtered)==0):
                continue
            
            # img_filtered_norm = (img_filtered + abs(np.min(img_filtered)))/(np.max(img_filtered) + abs(np.min(img_filtered))) #normalizing
            # thresh =  np.percentile(img_filtered_norm[img_filtered_norm > 0], 42) #produce threshold

            cysts_im = img_filtered <= thresh_hu
            cysts_im = 1*binary_closing(cysts_im)
            cysts_im *= mask
            cysts_im = medfilt(cysts_im)
            
            #need to navigate to correct directory 
            os.chdir(target_dir)
            new_scan_path = os.path.join(target_dir,list_of_scans[scan_num])
            if(not os.path.isdir(new_scan_path)):
                os.mkdir(new_scan_path)

            if(mask_path[-6:-4].isdigit()):
                new_slice_name = "cyst_mask_" + mask_path[-6:-4] + ".png"
            elif (mask_path[-5:-4].isdigit()):
                new_slice_name = "cyst_mask_0" + mask_path[-5:-4] + ".png"


            new_slice_path = os.path.join(new_scan_path, new_slice_name)

            
            pngfile = open(new_slice_path, 'wb') 
            imsave(pngfile, cysts_im)
            pngfile.close()




root_train_dir = "/Users/magdy/Desktop/Stanford Spring/BMI260/Project/Data/Cystic Dataset/Train"
target_train_dir = "/Users/magdy/Desktop/Stanford Spring/BMI260/Project/Data/Cystic_masks_new/Train"

root_test_dir = "/Users/magdy/Desktop/Stanford Spring/BMI260/Project/Data/Cystic Dataset/Test"
target_test_dir = "/Users/magdy/Desktop/Stanford Spring/BMI260/Project/Data/Cystic_masks_new/Test"

root_dir = root_train_dir
target_dir = target_train_dir

print("Generating Training Masks")
generate_cyst_mask(root_dir, target_dir, -916)

root_dir = root_test_dir
target_dir = target_test_dir

# print("Generating Test Masks")
# generate_cyst_mask(root_dir, target_dir)