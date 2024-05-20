import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd


class MRI_Generator(Dataset):

    def __init__(self,
                 data_path,
                 csv_path):

        self.data_path = data_path  # mris directory
        self.csv_path = csv_path
        self.dataframe = self.put_labels()

    def put_labels(self):

        try:
            df = pd.read_csv(self.csv_path)  # read the csv without specifying a separator
        except pd.errors.ParserError:
            df = pd.read_csv(self.csv_path, sep='\t')  # read with tab separator
        except Exception as e:
            raise Exception(f"An error occurred while reading the CSV file: {e}")

        assert 'mri' in df.columns
        assert 'group' in df.columns

        print('csv file read successfully !')

        # put labels - fcd group gets label 0 (epilepsy), hc group gets label 1 (non-epilepsy)
        list_labels = []
        for i in range(len(df)):
            if df.iloc[i]['group'] == 'fcd':
                list_labels.append(0)
            else:
                list_labels.append(1)
        df_labels = pd.DataFrame(list_labels, columns=['label'])
        df = pd.concat([df, df_labels], axis=1)
        # print(df)

        return df

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):

        mri_file = os.listdir(self.data_path)[idx]
        mri_path = os.path.join(self.data_path, mri_file)

        # sub = self.dataframe.loc[self.dataframe['mri'] == str(mri_file)[:-4], 'mri'].values[0] #works ok, correct sub-label assignment
        resized_map = torch.from_numpy(nib.load(mri_path).get_fdata())
        label = self.dataframe.loc[self.dataframe['mri'] == str(mri_file)[:-4], 'label'].values[0]  # [:-4] to skip the '.nii' extension
                                                                                                    # should be ok for all '.nii' files

        return torch.unsqueeze(resized_map, 0), torch.tensor(label)
