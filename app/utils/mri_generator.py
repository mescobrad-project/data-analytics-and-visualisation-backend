import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MRI_Generator(Dataset):

    def __init__(self,
                 dataset_participant_ids,
                 dataset_binary_labels,
                 data_path):

        self.dataset = dataset_participant_ids
        self.dataset_binary_labels = dataset_binary_labels
        self.data_path = data_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx): #to access mris via index

        participant = self.dataset[idx]
        path_participant = self.data_path + participant + '/anat/'
        s = [f for f in os.listdir(path_participant) if f.endswith('FLAIR.nii.gz')]
        path_new = path_participant + s[0]
        a = nib.load(path_new)
        a = a.get_fdata() #numpy.ndarray
        resized_map = torch.from_numpy(a) #torch array

        labels_binary = self.dataset_binary_labels[idx]

        #return the mri as torch array and its label
        return torch.unsqueeze(resized_map, 0), torch.from_numpy(np.array(labels_binary))