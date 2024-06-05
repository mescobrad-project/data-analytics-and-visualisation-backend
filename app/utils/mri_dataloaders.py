import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from app.utils.mri_generator import MRI_Generator

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y

def train_eval_dataloaders(data_path,
                           csv_path,
                           batch_size,
                           train_split=0.8):
    dataset = MRI_Generator(data_path, csv_path)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),  # Normalize to [0, 1]
        transforms.RandomHorizontalFlip(p=0.3),  # Apply random horizontal flipping with a probability of 0.3
        transforms.RandomVerticalFlip(p=0.3)
        #transforms.RandomRotation(20)
    ])

    # Apply the transformations
    transformed_dataset = TransformedDataset(dataset, transform)

    train_size = int(train_split * len(transformed_dataset))
    val_size = len(transformed_dataset) - train_size

    train_dataset, val_dataset = random_split(transformed_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('points in the dataloaders ', len(train_dataloader.dataset), len(val_dataloader.dataset))

    return train_dataloader, val_dataloader


def test_dataloader(data_path,
                    csv_path,
                    batch_size):

    dataset = MRI_Generator(data_path, csv_path)

    transform = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),  # Normalize to [0, 1]

    transformed_dataset = TransformedDataset(dataset, transform)

    test_dataloader = DataLoader(transformed_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return test_dataloader
