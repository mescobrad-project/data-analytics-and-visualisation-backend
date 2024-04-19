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

def visualize_dl(model_path,
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
    nii_img = nib.load(mri_path) #3-dim mri
    mri = nii_img.get_fdata()
    tensor_mri = torch.from_numpy(mri)
    tensor_mri = torch.unsqueeze(tensor_mri, 0)
    tensor_mri = torch.unsqueeze(tensor_mri, 0) #5-dim torch Tensor [1,1,160,256,256] (verified)

    print(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice,
                 alpha)

    return model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice,
                 alpha

    '''
    #--send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    tensor_mri = tensor_mri.to(device)
    print('tensor_mri', tensor_mri.shape)
    #model = model.to(device)
    wrapped_model.to(device)
    print(wrapped_model)

    #--deeplift
    dl = DeepLift(wrapped_model)
    output = model(tensor_mri)
    target_class = int(torch.argmax(output[1])) #int: this should be int 0 or 1 (verified)
    print('target class', target_class)

    # Track GPU memory usage
    initial_memory = torch.cuda.memory_allocated(device)
    print('Initial GPU Memory Allocated:', initial_memory)

    attributions = dl.attribute(tensor_mri, target=target_class)

    final_memory = torch.cuda.memory_allocated(device)
    max_memory = torch.cuda.max_memory_allocated(device)
    print('Attributions calculated! Shape:', attributions.shape)
    print('Final GPU Memory Allocated:', final_memory)
    print('Max GPU Memory Allocated:', max_memory)

    #--plot
    heatmap = attributions.cpu().squeeze().squeeze().permute(1, 2, 0).numpy() #[256, 256, 160] numpy array (verified)
    print('heatmap calculated! shape is', heatmap.shape)

    print('--- plot starts here! ---')
    fig, ax = plt.subplots(1, figsize=(6, 6))

    #cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
    #                                                 colors=[(1, 0, 0, 0), "darkred", "red", "darkorange", "orange", "yellow"],
    #                                                 N=5000)

    mri_array = tensor_mri.cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    print('mri_array for plot', mri_array.shape) #[256, 256, 160] torch Tensor (to verify)

    #pickle
    #with open(os.path.join(heatmap_path, 'mri_and_heatmap.pickle'), 'wb') as f:
    #    pickle.dump([mri_array, heatmap], f)

    ax.imshow(mri_array[:, :, slice], cmap="Greys")
    #im = ax.imshow(heatmap[:, :, slice], cmap=cmap, interpolation="gaussian", alpha=1)
    im = ax.imshow(heatmap[:, :, slice], alpha=alpha)
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
