import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from app.utils.mri_generator import MRI_Generator

def train_eval_dataloaders(data_path,
                           csv_path,
                           batch_size=5,
                           train_split=0.8):
    dataset = MRI_Generator(data_path, csv_path)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('points in the dataloaders ', len(train_dataloader.dataset), len(val_dataloader.dataset))

    return train_dataloader, val_dataloader


def test_dataloader(data_path,
                    csv_path,
                    batch_size):

    dataset = MRI_Generator(data_path, csv_path)
    test_dataloader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return test_dataloader
