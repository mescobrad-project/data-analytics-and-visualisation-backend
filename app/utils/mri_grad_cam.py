import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
#from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import pickle
from pytorch_grad_cam import GradCAM

# source is my 17.4(1) ntbk (xai, GD)
class Conv3DWrapper(nn.Module):
    def __init__(self, external_model):
        super(Conv3DWrapper, self).__init__()
        self.conv3d_model = external_model
    def forward(self, x):
        _, logits = self.conv3d_model(x)
        return logits

def visualize_grad_cam(model_path,
                       mri_path,
                       heatmap_path,
                       heatmap_name,
                       slice,
                       alpha):

    assert (os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

    model = torch.load(model_path)
    model.eval()

    nii_img = nib.load(mri_path)  # 3-dim mri
    mri = nii_img.get_fdata()
    tensor_mri = torch.from_numpy(mri)  # [160, 256, 256]
    tensor_mri = torch.unsqueeze(tensor_mri, 0)
    tensor_mri = torch.unsqueeze(tensor_mri, 0)  # 5-dim torch Tensor [1,1,160,256,256] (verified)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    model = model.to(device)
    wrapper_model = Conv3DWrapper(model)

    cam_instance = GradCAM(model=wrapper_model,
                           target_layers=[wrapper_model.conv3d_model.group5[0]])
    pixel_attributions = cam_instance(input_tensor=tensor_mri)[0, :,
                         :]  # for top predicted class - [256, 256, 160] numpy

    fig, ax = plt.subplots(1, figsize=(6, 6))

    # cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared', colors=[(1, 0, 0, 0), "darkred", "red", "darkorange", "orange", "yellow"], N=5000)

    mri_array = tensor_mri.cpu().squeeze().squeeze().permute(1, 2, 0).numpy()

    # pickle
    # with open(os.path.join(heatmap_path, 'mri_and_heatmap.pickle'), 'wb') as f:
    #    pickle.dump([mri_array, heatmap], f)

    ax.imshow(mri_array[:, :, slice], cmap="Greys")
    # im = ax.imshow(pixel_attributions[:, :, slice], cmap=cmap, interpolation="gaussian", alpha=1)
    im = ax.imshow(pixel_attributions[:, :, slice], interpolation="gaussian", alpha=alpha)
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
