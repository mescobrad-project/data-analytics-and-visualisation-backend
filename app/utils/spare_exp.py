import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn

def order_coronal_attributions(attributions):
    # attributions are numpy arrays of shape (256, 256, 160)

    attributions = np.where(attributions > 0, attributions, 0)  # consider only positive effect

    slice_attribution_sum = []
    for i in range(attributions.shape[1]):
        slice_attribution_sum.append(np.sum(attributions[:, i, :]))
    slice_attribution_sum_array = np.array(slice_attribution_sum)

    sorted_indices = np.argsort(slice_attribution_sum_array)[::-1]  # Get the indices that would sort the array in descending order

    return sorted_indices.tolist()

class MoRF_3D():

    def __init__(self,
                 mri_path,
                 attributions_path,
                 model_path):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mri = nib.load(mri_path).get_fdata()
        self.mri = torch.from_numpy(self.mri).unsqueeze(0).unsqueeze(0)
        self.mri = (self.mri - self.mri.min()) / (self.mri.max() - self.mri.min())  # mri values to [0,1] range
        self.mri = self.mri.to(device=self.device, dtype=torch.float32)
        print('mri: ', self.mri.min(), self.mri.max(), self.mri.shape)

        self.attributions = np.load(attributions_path)
        self.attributions = torch.from_numpy(self.attributions).to(device=self.device)
        print('attributions: ', self.attributions.min(), self.attributions.max(), self.attributions.shape)

        self.model = torch.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print('mri, attributions, model loaded ok')

    def perturbations(self,
                      plot_morf_curve=True,
                      plot_cumulative_differences=True):

        softmax = nn.Softmax(dim=1)
        perturbations = []

        with torch.no_grad():
            print('right after torch no grad')

            raw_scores = self.model(self.mri)[1]
            print('raw scores: ', raw_scores)
            probs = softmax(raw_scores)
            print('probs: ', probs)
            index = torch.argmax(probs)  # of top predicted class
            print('index: ', index)
            class_prob = round(float(probs[0, index]), 3)
            print('class_prob', class_prob)
            perturbations.append(class_prob)

            noise = torch.rand(256, 256, 160, device=self.device)  # random noise tensor with values in [0,1]
            print('noise', noise.min(), noise.max(), noise.shape)

            slices = order_coronal_attributions(self.attributions.cpu().numpy()) #indices of attributions array for the 2nd dim
            print('slices', min(slices), max(slices), len(slices))
            print('order_coronal_attributions calculated')

            for slice in slices:
                print('perturbations begin here')
                print('mri', self.mri.shape)
                print('mri shape for perturb', self.mri[0, 0, :, slice, :].shape)
                print('noise shape for perturb',  noise[:, slice, :].shape)
                self.mri[0, 0, :, slice, :] = noise[:, slice, :]
                raw_scores = self.model(self.mri)[1]
                perturbed_probs = softmax(raw_scores)
                class_perturbed_prob = round(float(perturbed_probs[0, index]), 3)
                perturbations.append(class_perturbed_prob)
                print(f'slice: {slice} - prob: {class_perturbed_prob}')

        if plot_morf_curve:
            plt.plot(range(0, len(perturbations)), perturbations)
            plt.title('MoRF Perturbation Curve')
            plt.xlabel('Perturbation steps')
            plt.ylabel('Predicted probability')
            plt.savefig(os.path.join(os.path.dirname(model_path), 'morf.png'))
            plt.show()

        differences = [perturbations[0] - perturbations[i] for i in range(1, len(perturbations))]
        L = len(self.attributions)
        score = (1 / (L + 1)) * sum(differences)
        print(f'aopc score: {round(score, 3)}')

        if plot_cumulative_differences:
            sum_of_differences = np.cumsum(differences)  # cumulative sum of differences
            plt.plot(range(1, len(perturbations)), sum_of_differences)
            plt.title('Cumulative differences')
            plt.xlabel('Perturbation steps')
            plt.ylabel('Sum of differences')
            plt.savefig(os.path.join(os.path.dirname(model_path), 'difs.png'))
            plt.show()

        return True

def get_3d_score(mri_path,
                 attributions_path,
                 model_path):
    
    MoRF_3D(mri_path, attributions_path, model_path).perturbations()

    return True
