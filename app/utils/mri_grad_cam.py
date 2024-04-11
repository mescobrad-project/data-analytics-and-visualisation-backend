import os
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
from PIL import Image

def visualize_grad_cam(model_path,
                       mri_path,
                       heatmap_path,
                       heatmap_name):
    
    model = torch.load(model_path)

    nii_img = nib.load(mri_path)
    mri = nii_img.get_fdata()
    tensor_mri = torch.tensor(mri)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    model = model.to(device)
    
    heatmap = grad_cam_heatmap(model, 
                               tensor_mri.unsqueeze(0).unsqueeze(0))

    #save heatmap as a png file in the heatmap_path
    heatmap = np.uint8(255 * heatmap)  # Convert to uint8 for saving as an image
    heatmap_img = Image.fromarray(heatmap, 'L')  # Create PIL image
    heatmap_img.save(os.path.join(heatmap_path, heatmap_name))

    return True

def grad_cam_heatmap(model, 
                     input_tensor):

    model.eval()

    # Forward pass
    output = model(input_tensor) #output = None, logits
    
    if output[1][0] > output[1][1]:
        target = torch.tensor([[1, 0]])
    else:
        target = torch.tensor([[0, 1]])

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output[1], target)

    # Backward pass
    model.zero_grad()
    loss.backward(retain_graph=True)

    # Get the gradients of the output with respect to the last conv layer (conv-batchnorm-relu-maxpool)
    gradients = model.group5[0].weight.grad
    pooled_gradients = torch.mean(gradients, dim=(2, 3, 4))  # Average pooling over spatial dimensions

    # Get the activations of the last convolutional layer
    activations = model.group1[0](input_tensor.float())  # Assuming the last convolutional layer is model.group1[0]
    activations = activations.detach()

    # Weight the channels by their importance in the gradient
    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :, :] *= pooled_gradients[i, :].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # ReLU on the heatmap
    heatmap = torch.relu(heatmap)

    # Normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Resize heatmap to the size of the input image
    heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0),
                                               size=(input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]),
                                               mode='trilinear', align_corners=False).squeeze()

    # Convert heatmap to numpy array
    heatmap = heatmap.detach().cpu().numpy()

    return heatmap
