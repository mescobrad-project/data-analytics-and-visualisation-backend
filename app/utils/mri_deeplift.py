import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from captum.attr import DeepLift
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

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

def resize_mri(mri, target_shape=(256, 256)):
    resized_slices = []
    for slice in mri:
        resized_slices.append(np.resize(slice, target_shape))
    return np.array(resized_slices)

def visualize_dl(model_path,
                 mri_path,
                 heatmap_path):

    assert os.path.exists(model_path)
    assert os.path.exists(mri_path)
    assert os.path.exists(heatmap_path)
    #assert axis in ['Sagittal', 'Coronal', 'Axial']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.load(model_path, map_location=device)
    wrapped_model = Conv3DWrapper(model)

    # Load MRI
    nparray = nib.load(mri_path).get_fdata()
    nparray = resize_mri(nparray) #
    tensor_mri = torch.from_numpy(nparray).unsqueeze(0).unsqueeze(0)
    tensor_mri = normalize(tensor_mri)
    tensor_mri = tensor_mri.to(device, dtype=torch.float32)

    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensor_mri = tensor_mri.to(device, dtype=torch.float32)
    # wrapped_model.to(device)

    # Prediction
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        logits = model(tensor_mri)[1]
    _, top_class = torch.max(softmax(logits), dim=1)
    group = 'Epilepsy' if top_class.item() == 0 else 'Non-Epilepsy'
    print(f'prediction: {group}')

    # DeepLift
    dl = DeepLift(wrapped_model)
    attributions = dl.attribute(tensor_mri, target=top_class.item()).detach().cpu().squeeze().permute(1, 2, 0).numpy()
    #attributions shape is (256, 256, 160) & they have both negative and positive values (thus normalization to [0,1] is needed for plot)
    #nib.save(nib.Nifti1Image(attributions, affine=np.eye(4)), os.path.join(heatmap_path, 'attributions.nii')) #save an nii
    #np.save(os.path.join(heatmap_path, 'attributions.npy'), attributions) #save as numpy array

    torch.cuda.empty_cache()

    # Prepare data for plotting
    attributions = normalize(attributions)
    tensor_mri = tensor_mri.detach().cpu().squeeze().permute(1, 2, 0).numpy() #convert mri to numpy array of shape (256, 256, 160)

    for i in range(attributions.shape[0]):
        mri_slice = tensor_mri[i, :, :]
        attr_slice = attributions[i, :, :]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mri_slice, cmap='Greys')
        sorted_values = np.sort(attr_slice.flatten())[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01) - 1]
        ax.imshow(np.where(attr_slice > threshold, attr_slice, 0),
                  cmap=LinearSegmentedColormap.from_list(name='blues',
                                                         colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"],
                                                         N=5000),
                  interpolation='gaussian')

        ax.set_title(f'Prediction: {group} \n\n MRI(Grey) vs DeepLift Attributions(Blue) Overlay \n Axial plane no. {i+1}')

        # Save and show plot
        heatmap_name = f'heatmap-axial-{i+1}.png'
        plt.savefig(os.path.join(heatmap_path, heatmap_name))
        plt.show()

    for i in range(attributions.shape[1]):
        mri_slice = tensor_mri[:, i, :]
        attr_slice = attributions[:, i, :]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mri_slice, cmap='Greys')
        sorted_values = np.sort(attr_slice.flatten())[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01) - 1]
        ax.imshow(np.where(attr_slice > threshold, attr_slice, 0),
                  cmap=LinearSegmentedColormap.from_list(name='blues',
                                                         colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"],
                                                         N=5000),
                  interpolation='gaussian')

        ax.set_title(f'Prediction: {group} \n\n MRI(Grey) vs DeepLift Attributions(Blue) Overlay \n Coronal plane no. {i+1}')

        # Save and show plot
        heatmap_name = f'heatmap-coronal-{i+1}.png'
        plt.savefig(os.path.join(heatmap_path, heatmap_name))
        plt.show()

    for i in range(attributions.shape[2]):
        mri_slice = tensor_mri[:,:,i]
        attr_slice = attributions[:,:,i]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mri_slice, cmap='Greys')
        sorted_values = np.sort(attr_slice.flatten())[::-1]
        threshold = sorted_values[int(tensor_mri.shape[0] * tensor_mri.shape[1] * 0.01) - 1]
        ax.imshow(np.where(attr_slice > threshold, attr_slice, 0),
                  cmap=LinearSegmentedColormap.from_list(name='blues',
                                                         colors=[(1, 0, 0, 0), "blue", "blue", "blue", "blue", "blue"],
                                                         N=5000),
                  interpolation='gaussian')

        ax.set_title(f'Prediction: {group} \n\n MRI(Grey) vs DeepLift Attributions(Blue) Overlay \n Sagittal plane no. {i+1}')

        # Save and show plot
        heatmap_name = f'heatmap-sagittal-{i+1}.png'
        plt.savefig(os.path.join(heatmap_path, heatmap_name))
        plt.show()
    
    return True


# path = NeurodesktopStorageLocation + "/model_data/saved_models_2024-06-12_17-47/"
# model_path = path + "conv3d_experiment1.pth"
# mri_path = path + "mris_test/sub-00060.nii"
# heatmap_path = path
# visualize_dl(model_path,
#              mri_path,
#              heatmap_path)
# yields RuntimeError: .. not enough memory ...
