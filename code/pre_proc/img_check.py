### -- SCRIPT TO CHECK IF ALL IMAGING DATA IS ZERO OR NOT

import nibabel as nib
import os
import numpy as np

folder = '5_T1_MRI/ImageFeature/LogJacobian'  # replace with your path
# all_zero_files = []

# for filename in os.listdir(folder):
#     path = os.path.join(folder, filename)
#     img = nib.load(path)
#     data = img.get_fdata()
#     if np.all(data == 0):
#         all_zero_files.append(filename)

# print(f"{len(all_zero_files)} files with all zeros")


# with open('empty_images.txt', 'w') as f:
#     for f in all_zero_files:
#         f.write(f"{f}\n")

## -- look at the data from a specifc ID (ID = 289)

img_check = nib.load(os.path.join(folder, "0003.nii.gz"))
data = img_check.get_fdata()
# Get shape and initialize result list
_, _, d3 = data.shape
zero_counts = []
na_counts = []

# Count zeros in each slice along the 3rd dimension
for i in range(d3):
    slice_i = data[:, :, i]
    zero_count = np.sum(slice_i == 0)
    zero_counts.append(zero_count)

for i, count in enumerate(zero_counts):
    print(f"Slice {i}: {round(count / (216*256), 3)}%")

# Count NAs in each slice along the 3rd dimension
for i in range(d3):
    slice_i = data[:, :, i]
    na_count = np.sum(np.isnan(slice_i))
    na_counts.append(na_count)

for i, count in enumerate(na_counts):
    print(f"Slice {i}: {round(count / (216*256), 3)}%")
