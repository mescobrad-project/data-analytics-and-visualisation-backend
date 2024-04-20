import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from captum.attr import DeepLift
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

class Conv3DWrapper(nn.Module):
    def __init__(self, external_model):
        super(Conv3DWrapper, self).__init__()
        self.conv3d_model = external_model
    def forward(self, x):
        _, logits = self.conv3d_model(x)
        return logits

def normalize(input):
    # input can be 2-dim or 3-dim
    min = input.min()
    max = input.max()
    normalized = (input - min) / (max - min)
    return normalized

def visualize_dl(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice):

    assert(os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

    #--load model
    model = torch.load(model_path)
    wrapped_model = Conv3DWrapper(model)

    #--load mri
    mri = nib.load(mri_path).get_fdata()
    tensor_mri = torch.from_numpy(mri)
    tensor_mri = tensor_mri.unsqueeze(0).unsqueeze(0) #5-dim torch Tensor [1,1,160,256,256] (verified)

    #--send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))
    tensor_mri = tensor_mri.to(device)
    wrapped_model.to(device)

    #--deeplift
    dl = DeepLift(wrapped_model)
    target_class = int(torch.argmax(model(tensor_mri)[1])) #int: this should be int 0 or 1 (verified)
    print('target class (model prediction): ', target_class)
    # Track GPU memory usage
    print('Initial GPU Memory Allocated:', torch.cuda.memory_allocated(device)) #almost 11.72GB
    #print('GPU Summary:', torch.cuda.memory_summary())
    attributions = dl.attribute(tensor_mri, target=target_class)
    print('Attributions calculated! Shape:', attributions.shape)
    print('Final GPU Memory Allocated:', torch.cuda.memory_allocated(device))
    print('Max GPU Memory Allocated:', torch.cuda.max_memory_allocated(device))

    #--plots
    attributions = attributions.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()  # [256, 256, 160] numpy array (verified)
    tensor_mri = tensor_mri.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    #with open(os.path.join(heatmap_path, 'mri_and_heatmap.pickle'), 'wb') as f:
    #     pickle.dump([tensor_mri, atsutributions], f)

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))

    # Plot MRI
    img1 = ax[0].imshow(normalize(tensor_mri[:, :, slice]), cmap='Greys')
    ax[0].set_title('MRI - Slice {}'.format(slice))
    fig.colorbar(img1, ax=ax[0])

    # Plot attributions
    img2 = ax[1].imshow(normalize(attributions[:, :, slice]), cmap='viridis')
    ax[1].set_title('Deeplift Attributions - Slice {}'.format(slice))
    fig.colorbar(img2, ax=ax[1])

    # Plot overlay
    ax[2].imshow(normalize(tensor_mri[:, :, slice]), cmap='Greys')
    cmap = mcolors.LinearSegmentedColormap.from_list(name='blues',
                                                     colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"],
                                                     N=5000)
    #slight adjustment to drop low importance values, as they create fuzzy and confusing regions on the mri slice
    sorted_values = np.sort(normalize(attributions[:, :, slice].flatten()))[::-1]
    threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01)-1] # 1% of total slice pixels
    ax[2].imshow(np.where(normalize(attributions[:, :, slice]) > threshold, normalize(attributions[:, :, slice]), 0),
                 cmap=cmap,
                 interpolation='gaussian')
    ax[2].set_title('MRI(Greyscale) vs Attributions(Blue) Overlay - Slice {}'.format(slice))

    # Save and show the plot
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
