### -- EDA ON T1 MRI DATA

### -- Pre-processing needs to be done on the T1 MRI data that we have. 
  ## - It would also be nice to do some initial EDA on this image data.

### -- LOAD LIBRARIES

import pandas as pd
import os
import matplotlib as plt
import seaborn as sns
import polars as pl
from pathlib import Path
import nibabel as nib
import numpy as np
import re
import random
import torch
import csv

### -- MAIN INPUTS

path_img = '5_T1_MRI/ImageFeature/'
path_tab = '5_T1_MRI/TabularFeature/'

### -- TABULAR FEATURES

file_names = ["Global", "ROI_Thick", "ROI_VolumeInVoxels"]

for file in file_names:
    print(f"--- Head of {file} ---")
    df = pl.read_csv("5_T1_MRI/TabularFeature/" + file + ".csv")
    print(df.shape)
    print(df.head())
    print()


#### -- Seems like tabular features are available for 1336 subjects (but imaging only for 1268)

#### -- Are all the IDs the same across all three datasets?

ids = [pl.read_csv("5_T1_MRI/TabularFeature/" + file + ".csv").get_column("ID") for file in file_names]

#### Check if all are equal to the first one
all_same = all((id == ids[0]).all() for id in ids[1:])

print("IDs are the same across all datasets:" , all_same)

#### -- do all IDs which have clinical info have imaging/tabular data?

df_clinical = pd.read_csv("1_Clinical/ADNIMERGE_01Oct2024.csv")

#### -- EDA on the tabular data

### -- deriving the number of cortical regions/vertices








### -- IMAGE FEATURES

#### -- let's first see how many files are in each of the main two folders (logJac, normBrain)

def count_files(folder_path):
    return sum(1 for f in Path(folder_path).iterdir() if f.is_file())

folder1 = Path('5_T1_MRI/ImageFeature/LogJacobian')
folder2 = Path('5_T1_MRI/ImageFeature/NormalizedBrain')

print(f"LogJacobian has {count_files(folder1)} files.")
print(f"Normalized 2 has {count_files(folder2)} files.")


#### -- try to read one MRI file

img = nib.load(path_img + 'LogJacobian/0004.nii.gz')

# Get the data as a NumPy array
data = img.get_fdata()

# Print basic info
print("Data shape:", data.shape)
print("Voxel size (pixdim):", img.header.get_zooms())
print("Affine matrix:", img.affine)

### -- check if all MRI files have the same shape

def same_size(folder_path):
  shapes = {}

  for file in os.listdir(folder_path):
      if file.endswith('.nii.gz'):
          path = os.path.join(folder_path, file)
          img = nib.load(path)
          shape = img.shape
          if shape not in shapes:
              shapes[shape] = []
          shapes[shape].append(file)

  # Print summary of shapes
  for shape, files in shapes.items():
      print(f"Shape {shape}: {len(files)} files")

  # Optional: check if only one shape
  if len(shapes) == 1:
      print("✅ All files have the same shape.")
  else:
      print("⚠️ Found multiple shapes:")

logJac = path_img + 'LogJacobian/'
normBrain = path_img + 'NormalizedBrain/'

same_size(logJac)
same_size(normBrain)


#### -- these are available for only 1268 subject (compared to 1336 tabular subjects). 
# subset tabular to only those IDs which are common to both.

directory = "5_T1_MRI/ImageFeature/LogJacobian"
files = os.listdir(directory)

image_ids = []
for f in files:
    match = re.findall(r'\d+', f)
    if match:
        image_ids.extend([int(x) for x in match])

tabular_ids = (
    pl.read_csv("5_T1_MRI/TabularFeature/Global.csv")
      .select("ID")
      .to_series()
      .to_list()
)

common_ids = list(set(tabular_ids) & set(image_ids))
img_tab = len(common_ids)
print(f"ID count common to both tabular and imaging: {img_tab} files")

for file in file_names:
    df = pd.read_csv("5_T1_MRI/TabularFeature/" + file + ".csv")
    
    # Subset based on ID
    df_subset = df[df["ID"].isin(image_ids)]
    print(df_subset.shape)
    
    # Save or print
    output_path = os.path.join("5_T1_MRI/TabularFeature", f"{file}_v2.csv")
    df_subset.to_csv(output_path, index=False)
    print(f"Subset saved to: {output_path}")

## -- write the common IDs to a CSV (cause will need this to susbet to demographics)
common_df = pd.DataFrame(common_ids, columns=['ID'])
common_df.to_csv('common_ids.csv', index=False)


### -- LOOK AT WHAT THE IMAGING DATA LOOKS LIKE
output_csv = "eda/image_stats.csv"

# Prepare header
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'mean', 'std'])

    for id in common_ids:
        img = nib.load(path_img + 'LogJacobian/' + str(id).zfill(4) + '.nii.gz')

        # Get the data as a NumPy array
        data = img.get_fdata().astype(np.float32)
        img_mean = np.mean(data)
        img_std = np.std(data)
        
        # Write row
        writer.writerow([id, img_mean, img_std])






