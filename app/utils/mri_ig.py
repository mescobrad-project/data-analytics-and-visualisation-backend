import os
import numpy as np
import torch
import nibabel as nib
from captum.attr import IntegratedGradients
from PIL import Image


def visualize_ig(model_path,
                 mri_path,
                 heatmap_path,
                 heatmap_name):

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

    return True
