# @title mri_explanations.py

import numpy as np
from copy import deepcopy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib

from app.utils.lrp_investigator import InnvestigateModel

def lrp_explanation(model_path,
                    mri_path,
                    mri_slice,
                    output_file_path,
                    label=None,
                    vmin=90, vmax=99.9):
  
    model = torch.load(model_path)

    nii_img = nib.load(mri_path)
    mri = nii_img.get_fdata()
    tensor_mri = torch.tensor(mri)

    #convert mri to 5-dim tensor, ex torch.Size([1, 1, 157, 256, 256])
    if len(tensor_mri.shape) == 3:
        tensor_mri = tensor_mri.unsqueeze(0).unsqueeze(0)
    elif len(tensor_mri.shape) == 4:
        tensor_mri = tensor_mri.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mri = tensor_mri.to(device)
    model = model.to(device)

    if label:
        _, logits = model(tensor_mri)
    else:
        logits = model(tensor_mri)

    classes = ['FCD', 'HC']

    if label is None:
        target = classes[int(torch.argmax(logits[0]))]
        print(f'No label is provided. Target at class {target}.')
    else:
        target = classes[int(label)]
        if int(label) == int(torch.argmax(logits[0])):
            print(f'Correct prediction. Target at class {target}.')
        else:
            print(f'Wrong prediction. Target at class {target}.')

    inn_model = InnvestigateModel(model,
                                  lrp_exponent=2,
                                  method="e-rule",
                                  beta=0)

    model_prediction, heatmap = inn_model.innvestigate(in_tensor=tensor_mri) #heatmap of same size as the mri

    mean_maps_LRP = mean_maps(model_prediction, heatmap)
    ref_scale = mean_maps_LRP[target]

    shape = tensor_mri.shape
    x_idx=slice(0, shape[4])
    y_idx=slice(0, shape[3])
    assert 0<=mri_slice and mri_slice<=shape[2]
    z_idx=mri_slice

    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',
                                                  colors=[(1, 0, 0, 0),
                                                         "darkred", "red", "darkorange", "orange", "yellow"], N=5000)

    brain_img = tensor_mri[0,0,:,:,:].permute(1, 2, 0)
    ax.imshow(brain_img[x_idx, y_idx, z_idx].T, cmap="Greys")

    vmin_val, vmax_val = np.percentile(ref_scale, vmin), np.percentile(ref_scale, vmax)
    img = heatmap[0,0,:,:,:].permute(1, 2, 0)
    im = ax.imshow(img[x_idx, y_idx, z_idx].T, cmap=cmap,
               vmin=vmin_val, vmax=vmax_val, interpolation="gaussian")

    ax.set_title('Plot of slice no. {}'.format(z_idx))
    ax.axis('off')
    plt.gca().invert_yaxis()

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)

    plt.savefig(output_file_path, dpi=300)  # Set dpi for better quality if needed

    #cbar_ax = fig.add_axes([0.8, 0.15, 0.025, 0.7])
    #cbar = fig.colorbar(im, shrink=0.5, ticks=[vmin, vmax], cax=cbar_ax)
    #cbar.set_ticks([vmin_val, vmax_val])
    #cbar.ax.set_yticklabels(['{0:.1f}%'.format(vmin), '{0:.1f}%'.format(vmax)], fontsize=14)
    #cbar.set_label('Percentile of  average AD patient values\n', rotation=270, fontsize=14)

    return fig, ax

def mean_maps(model_prediction,
              heatmap):

    cases = ["FCD", "HC", "TP", "TN", "FP", "FN"]
    mean_maps_LRP = {case: np.zeros(heatmap.shape) for case in cases}
    counts = {case: 0 for case in cases}

    num_samples = 1

    ad_score_list = []

    AD_score, LRP_map = model_prediction, heatmap
    AD_score = AD_score[0][1].detach().cpu().numpy()
    LRP_map = LRP_map.detach().cpu().numpy().squeeze()
    ad_score_list.append(AD_score)

    label = torch.argmax(model_prediction)

    true_case = "FCD" if label else "HC"

    if AD_score.round() and label:
        case = "TP"
    elif AD_score.round() and not label:
        case = "FP"
    elif not AD_score.round() and label:
        case = "FN"
    elif not AD_score.round() and not label:
        case = "TN"

    mean_maps_LRP[case] += LRP_map
    counts[case] += 1
    mean_maps_LRP[true_case] += LRP_map
    counts[true_case] += 1

    return mean_maps_LRP
