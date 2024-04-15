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

    assert(os.path.exists(model_path))
    assert (os.path.exists(mri_path))
    assert (os.path.exists(heatmap_path))

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
        return model(inp1)[1]

    ig = IntegratedGradients(wrapped_model)

    output = model(tensor_mri)
    target_class = int(torch.argmax(output[1])) #this should be int 0 or 1

    print(tensor_mri.shape)
    print(target_class)
    attributions, approximation_error = ig.attribute(tensor_mri,
                                                     method='gausslegendre',
                                                     n_steps=4,
                                                     target=target_class,
                                                     return_convergence_delta=True)
    print('attributions', attributions.shape)

    heatmap = attributions.cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    print('heatmap', heatmap.shape)

    min_val = heatmap.min()
    max_val = heatmap.max()

    normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
    print('normalized_heatmap', normalized_heatmap.shape)

    #heatmap_img = Image.fromarray((normalized_heatmap[:, :, slice] * 255).astype(np.uint8))
    heatmap_img = normalized_heatmap[:, :, slice]
    heatmap_img.save(os.path.join(heatmap_path, heatmap_name))
    print('heatmap saved')

    # should include overlap here as well

    fig, ax = plt.subplots(1, figsize=(6, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
                                                     colors=[(1, 0, 0, 0), "darkred", "red", "darkorange", "orange", "yellow"],
                                                     N=5000)

    mri_array = tensor_mri.cpu().squeeze().squeeze().permute(1, 2, 0).numpy()
    print(mri_array.shape)

    ax.imshow(mri_array[slice, :, :], cmap="Greys")
    print('mri plotted')

    im = ax.imshow(heatmap_img,
                   cmap=cmap,
                   interpolation="gaussian",
                   alpha=1)
    print('heatmap plotted')

    plt.savefig(os.path.join(heatmap_path, heatmap_name+'overlay.jpg'))
    print('figure saved')

    plt.show()

    return True
