import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from captum.attr import DeepLift
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
#import torch.nn.functional as F

class Conv3DWrapper(nn.Module):
    def __init__(self, external_model):
        super(Conv3DWrapper, self).__init__()
        self.conv3d_model = external_model

    def forward(self, x):
        _, logits = self.conv3d_model(x)
        return logits


def normalize(input):
    min_val = input.min()
    max_val = input.max()
    return (input - min_val) / (max_val - min_val)


def visualize_dl(model_path, mri_path, heatmap_path, heatmap_name, axis, slice_idx):
    assert os.path.exists(model_path)
    assert os.path.exists(mri_path)
    assert os.path.exists(heatmap_path)
    assert axis in ['sagittal', 'frontal', 'axial']

    # Load model
    model = torch.load(model_path)
    wrapped_model = Conv3DWrapper(model)

    # Load MRI
    mri = nib.load(mri_path).get_fdata()
    tensor_mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    wrapped_model.to(device)

    # Prediction
    top_class = int(torch.argmax(model(tensor_mri)[1]))
    group = 'Epilepsy' if top_class == 0 else 'Non-Epilepsy'

    # DeepLift
    dl = DeepLift(wrapped_model)
    attributions = dl.attribute(tensor_mri, target=top_class).detach().cpu().squeeze().numpy()

    # Prepare data for plotting
    tensor_mri = tensor_mri.detach().cpu().squeeze().numpy()
    if axis == 'sagittal':
        mri_slice = tensor_mri[:, :, slice_idx]
        attr_slice = attributions[:, :, slice_idx]
    elif axis == 'frontal':
        mri_slice = tensor_mri[:, slice_idx, :]
        attr_slice = attributions[:, slice_idx, :]
    elif axis == 'axial':
        mri_slice = tensor_mri[slice_idx, :, :]
        attr_slice = attributions[slice_idx, :, :]

    # Normalize and create plot
    mri_slice = normalize(mri_slice)
    attr_slice = normalize(attr_slice)
    sorted_values = np.sort(attr_slice.flatten())[::-1]
    threshold = sorted_values[int(mri_slice.size * 0.01) - 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mri_slice, cmap='Greys')
    ax.imshow(np.where(attr_slice > threshold, attr_slice, 0),
              cmap=LinearSegmentedColormap.from_list('blues', [(1, 0, 0, 0), "blue"], N=5000),
              interpolation='gaussian')

    ax.set_title(f'MRI(Grey) vs DeepLift Attributions(Blue) Overlay\npred: {group}\n{axis} slice {slice_idx}')

    # Save and show plot
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True

'''
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
                 axis,
                 slice):

    assert(os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

    assert axis in ['sagittal', 'frontal', 'axial']

    #--MODEL
    model = torch.load(model_path)
    wrapped_model = Conv3DWrapper(model)

    #--MRI
    mri = nib.load(mri_path).get_fdata()
    tensor_mri = torch.from_numpy(mri)
    tensor_mri = tensor_mri.unsqueeze(0).unsqueeze(0) #5-dim torch Tensor [1,1,160,256,256] (verified)

    #--DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))
    tensor_mri = tensor_mri.to(device)
    wrapped_model.to(device)

    #--PREDICTION
    print(model(tensor_mri)[1])
    top_class = int(torch.argmax(model(tensor_mri)[1]))
    #top_prob, top_class = torch.max(F.softmax(wrapped_model(tensor_mri), dim=1), dim=1)
    if top_class == 0:
        group = 'Epilepsy' #(fcd)
    elif top_class == 1:
        group = 'Non-Epilepsy' #(hc)

    #--DEEPLIFT
    dl = DeepLift(wrapped_model)
    # Track GPU memory usage
    #print('Initial GPU Memory Allocated:', torch.cuda.memory_allocated(device)) #almost 11.72GB
    #print('GPU Summary:', torch.cuda.memory_summary())
    attributions = dl.attribute(tensor_mri, target=top_class)
    #print('Attributions calculated! Shape:', attributions.shape)
    #print('Final GPU Memory Allocated:', torch.cuda.memory_allocated(device))
    #print('Max GPU Memory Allocated:', torch.cuda.max_memory_allocated(device))

    #--PLOTS - all are [spatial_x, spatial_y, height_volume] in the plot section
    attributions = attributions.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()  # [256, 256, 160] numpy array (verified)
    tensor_mri = tensor_mri.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot MRI
    #img1 = ax[0].imshow(normalize(tensor_mri[:, :, slice]), cmap='Greys')
    #if top_class == 0:
    #    ax[0].set_title('Epileptic MRI - Slice {}'.format(slice))
    #else:
    #    ax[0].set_title('Non-Epileptic MRI - Slice {}'.format(slice))
    #fig.colorbar(img1, ax=ax[0])

    # Plot attributions
    #img2 = ax[1].imshow(normalize(attributions[:, :, slice]),
    #                    cmap=LinearSegmentedColormap.from_list(name='yellow_to_blue',
    #                                                           colors=[(1, 1, 0), (0, 0, 1)]))
    #ax[1].set_title('Deeplift Attributions - Slice {}'.format(slice))
    #fig.colorbar(img2, ax=ax[1])

    # Plot overlay
    if axis == 'sagittal':
        ax.imshow(normalize(tensor_mri[:, :, slice]), cmap='Greys')
        sorted_values = np.sort(normalize(attributions[:, :, slice].flatten()))[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01)-1] # 1% of total slice pixels
        ax.imshow(np.where(normalize(attributions[:, :, slice]) > threshold, normalize(attributions[:, :, slice]), 0),
                     cmap=LinearSegmentedColormap.from_list(name='blues', colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"], N=5000),
                     interpolation='gaussian')
    elif axis == 'frontal':
        ax.imshow(normalize(tensor_mri[:, slice, :]), cmap='Greys')
        sorted_values = np.sort(normalize(attributions[:, slice, :].flatten()))[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01)-1] # 1% of total slice pixels
        ax.imshow(np.where(normalize(attributions[:, slice, :]) > threshold, normalize(attributions[:, slice, :]), 0),
                     cmap=LinearSegmentedColormap.from_list(name='blues', colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"], N=5000),
                     interpolation='gaussian')
    elif axis == 'axial':
        ax.imshow(normalize(tensor_mri[slice, :, :]), cmap='Greys')
        sorted_values = np.sort(normalize(attributions[slice, :, :].flatten()))[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01)-1] # 1% of total slice pixels
        ax.imshow(np.where(normalize(attributions[slice, :, :]) > threshold, normalize(attributions[slice, :, :]), 0),
                     cmap=LinearSegmentedColormap.from_list(name='blues', colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"], N=5000),
                     interpolation='gaussian')

    ax.set_title('MRI(Grey) vs DeepLift Attributions(Blue) Overlay\n' + f'pred: {group}\n' + f'{axis} slice {slice}')

    # Save and show the plot
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
'''
