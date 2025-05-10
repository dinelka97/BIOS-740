### -- SCRIPT TO SHOW IMAGES

import nibabel as nib
import matplotlib.pyplot as plt

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load image
img = nib.load('5_T1_MRI/ImageFeature/LogJacobian/0003.nii.gz')  # update path
data = img.get_fdata()

# Middle slice indices
sag_i = data.shape[0] // 2
cor_i = data.shape[1] // 2
axi_i = data.shape[2] // 2

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Sagittal
axs[0].imshow(np.rot90(data[sag_i, :, :]), cmap='gray')
axs[0].set_title('Sagittal slice')
axs[0].axis('off')

# Coronal
axs[1].imshow(np.rot90(data[:, cor_i, :]), cmap='gray')
axs[1].set_title('Coronal slice')
axs[1].axis('off')

# Axial
axs[2].imshow(np.rot90(data[:, :, axi_i]), cmap='gray')
axs[2].set_title('Axial slice')
axs[2].axis('off')

# Save to file
plt.tight_layout()
plt.savefig('brain_slices.png', dpi=150, bbox_inches='tight')
