import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from captum.attr import GuidedGradCam
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import torch.nn.functional as F

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

def visualize_ggc(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 axis,
                 slice):

    assert(os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

    assert axis in ['sagittal', 'frontal', 'axial']

    #--load model
    model = torch.load(model_path)
    wrapped_model = Conv3DWrapper(model)

    #--load mri
    tensor_mri = nib.load(mri_path).get_fdata()
    tensor_mri = torch.from_numpy(tensor_mri).unsqueeze(0).unsqueeze(0))

    #--send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    wrapped_model.to(device)

    # Prediction
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        logits = model(tensor_mri)[1]
    top_prob, top_class = torch.max(softmax(logits), dim=1)
    group = 'Epilepsy' if top_class.item() == 0 else 'Non-Epilepsy'

    # Guided Grad-CAM
    ggc = GuidedGradCam(wrapped_model, layer=wrapped_model.conv3d_model.group6)
    attributions = ggc.attribute(tensor_mri, target=top_class)
    attributions = attributions.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()  # [256, 256, 160] numpy array (verified)
    tensor_mri = tensor_mri.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()

    if axis == 'sagittal':
        mri_slice = tensor_mri[:, :, slice_idx]
        attr_slice = attributions[:, :, slice_idx]
    elif axis == 'frontal':
        mri_slice = tensor_mri[:, slice_idx, :]
        attr_slice = attributions[:, slice_idx, :]
    elif axis == 'axial':
        mri_slice = tensor_mri[slice_idx, :, :]
        attr_slice = attributions[slice_idx, :, :]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize(mri_slice), cmap='Greys')
    sorted_values = np.sort(normalize(attr_slice).flatten())[::-1]
    threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01) - 1]
    ax.imshow(np.where(normalize(attr_slice) > threshold, normalize(attr_slice), 0),
              cmap=LinearSegmentedColormap.from_list(name='blues',
                                                     colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"],
                                                     N=5000),
              interpolation='gaussian')

    ax.set_title(f'MRI(Grey) vs GuidedGC Attributions(Blue) Overlay\npred: {group} (prob: {round(top_prob.item(), 2)})\n{axis} slice {slice_idx}')

    # Save and show the plot
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
