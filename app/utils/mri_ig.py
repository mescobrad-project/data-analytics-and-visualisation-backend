import os
import numpy as np
import torch
import nibabel as nib
from captum.attr import IntegratedGradients
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_ig(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name,
                 slice):

    model = torch.load(model_path)

    nii_img = nib.load(mri_path)
    mri = nii_img.get_fdata()
    tensor_mri = torch.from_numpy(mri)
    tensor_mri = torch.unsqueeze(tensor_mri, 0)
    tensor_mri = torch.unsqueeze(tensor_mri, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    model = model.to(device)

    def wrapped_model(inp1):
        return model(inp1)[0]

    ig = IntegratedGradients(wrapped_model)

    output = model(tensor_mri)
    target_class = int(torch.argmax(output[1])) #this should be int 0 or 1

    attributions, approximation_error = ig.attribute(tensor_mri,
                                                     method='gausslegendre',
                                                     n_steps=4,
                                                     target=target_class,
                                                     return_convergence_delta=True)
    print(attributions.shape)

    heatmap = attributions.squeeze().squeeze().permute(1, 2, 0).numpy()

    min_val = heatmap.min()
    max_val = heatmap.max()
    normalized_heatmap = (heatmap - min_val) / (max_val - min_val)

    heatmap_img = Image.fromarray((normalized_heatmap * 255).astype(np.uint8))

    heatmap_img.save(os.path.join(heatmap_path, heatmap_name))

    # should include overlap here as well
    fig, ax = plt.subplots(1, figsize=(6, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
                                                     colors=[(1, 0, 0, 0),
                                                             "darkred", "red", "darkorange", "orange", "yellow"],
                                                     N=5000)
    ax.imshow(normalized_heatmap[slice, :, :], cmap="Greys")
    mri_array = tensor_mri.squeeze().squeeze().permute(1, 2, 0).numpy()
    im = ax.imshow(mri_array[256, 256, slice],
                   cmap=cmap,
                   interpolation="gaussian",
                   alpha=1)
    plt.savefig(heatmap_path)
    plt.show()

    return True
