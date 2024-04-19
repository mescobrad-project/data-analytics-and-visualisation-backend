import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from captum.attr import GuidedGradCam
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

def visualize_ggc(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice,
                 alpha):

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

    #--GGC
    ggc = GuidedGradCam(wrapped_model, layer=wrapped_model.conv3d_model.group5)
    target_class = int(torch.argmax(model(tensor_mri)[1])) #int: this should be int 0 or 1 (verified)
    print('target class (model prediction): ', target_class)

    # Track GPU memory usage
    print('Initial GPU Memory Allocated:', torch.cuda.memory_allocated(device)) #almost 11.72GB
    #print('GPU Summary:', torch.cuda.memory_summary())

    attributions = ggc.attribute(tensor_mri, target=target_class)

    print('Attributions calculated! Shape:', attributions.shape)
    print('Final GPU Memory Allocated:', torch.cuda.memory_allocated(device))
    print('Max GPU Memory Allocated:', torch.cuda.max_memory_allocated(device))

    #--plot
    attributions = attributions.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()  # [256, 256, 160] numpy array (verified)
    tensor_mri = tensor_mri.detach().cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    #with open(os.path.join(heatmap_path, 'mri_and_heatmap.pickle'), 'wb') as f: pickle.dump([tensor_mri, attributions], f)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot MRI
    ax[0].imshow(normalize(tensor_mri[:, :, slice]))
    ax[0].set_title('MRI slice {}'.format(slice))

    # Plot attributions
    ax[1].imshow(normalize(attributions[:, :, slice]))
    ax[1].set_title('Attributions slice {}'.format(slice))

    # Plot overlay
    cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
                                                     colors=[(1, 0, 0, 0), "darkred", "red", "darkorange", "orange", "yellow"],
                                                     N=5000)
    ax[2].imshow(normalize(tensor_mri[:, :, slice]), cmap="Greys")
    im = ax[2].imshow(normalize(attributions[:, :, slice]), cmap=cmap, interpolation="gaussian", alpha=alpha)
    ax[2].set_title('Overlay')

    # Save and show the plot
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
