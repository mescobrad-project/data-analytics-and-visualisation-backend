import os
import numpy as np
import torch
import nibabel as nib
from captum.attr import IntegratedGradients
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

def visualize_ig(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice):

    assert(os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

    model = torch.load(model_path)

    nii_img = nib.load(mri_path) #3-dim mri
    mri = nii_img.get_fdata()
    tensor_mri = torch.from_numpy(mri)
    tensor_mri = torch.unsqueeze(tensor_mri, 0)
    tensor_mri = torch.unsqueeze(tensor_mri, 0) #5-dim torch Tensor [1,1,160,256,256] (verified)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    model = model.to(device)

    def wrapped_model(inp1):
        return model(inp1)[1]

    ig = IntegratedGradients(wrapped_model)

    output = model(tensor_mri)
    target_class = int(torch.argmax(output[1])) #int: this should be int 0 or 1 (verified)

    attributions, approximation_error = ig.attribute(tensor_mri,
                                                     method='gausslegendre',
                                                     n_steps=4,
                                                     target=target_class,
                                                     return_convergence_delta=True)
    #print('attributions', attributions.shape) #5-dim torch Tensor [1,1,160,256,256] (verified)

    heatmap = attributions.cpu().squeeze().squeeze().permute(1, 2, 0).numpy() #[256, 256, 160] numpy array (verified)

    fig, ax = plt.subplots(1, figsize=(6, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
                                                     colors=[(1, 0, 0, 0), "darkred", "red", "darkorange", "orange", "yellow"],
                                                     N=5000)

    mri_array = tensor_mri.cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    print(mri_array.shape) #[256, 256, 160] torch Tensor (to verify)

    #pickle
    with open(os.path.join(heatmap_path, 'mri_and_heatmap.pickle'), 'wb') as f:
        pickle.dump([mri_array, heatmap], f)

    ax.imshow(mri_array[:, :, slice], cmap="Greys")
    im = ax.imshow(heatmap[:, :, slice], cmap=cmap, interpolation="gaussian", alpha=1)
    plt.savefig(os.path.join(heatmap_path, heatmap_name))
    plt.show()

    return True
