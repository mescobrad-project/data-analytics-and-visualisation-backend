import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


def train_val_test_sets(csv_path,
                        test_size):
    os.path.exists(csv_path)
    assert 0 < test_size < 1

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)

    y = np.array(df['label'])

    try:
        df_new = df.drop(['Unnamed: 0', 'label'], axis=1)
    except KeyError:
        df_new = df.drop(['label'], axis=1)

    # Normalize the data to the 0-1 range: training deteriorates, keep this transformation out for now
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(df_new)

    X = np.array(df_new)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    val_size = int(X_train.shape[0] * 0.1)  # to be set internally as 10% of the train data

    X_train_new, X_eval, y_train_new, y_eval = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train,
                                                                random_state=42)

    X_train_new = torch.Tensor(X_train_new)
    y_train_new = torch.Tensor(y_train_new)
    X_eval = torch.Tensor(X_eval)
    y_eval = torch.Tensor(y_eval)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # print(X_train_new.shape, y_train_new.shape, X_eval.shape, y_eval.shape, X_test.shape, y_test.shape)

    return X_train_new, y_train_new, X_eval, y_eval, X_test, y_test


class Generator(Dataset):

    def __init__(self, dataset_text, dataset_binary_labels):
        self.dataset_text = dataset_text
        self.dataset_binary_labels = dataset_binary_labels

    def __len__(self):
        return len(self.dataset_binary_labels)

    def __getitem__(self, idx):
        return self.dataset_text[idx], self.dataset_binary_labels[idx].float()


def dataloaders(csv_path, test_size):
    X_train_new, Y_train_new, X_eval, Y_eval, X_test, Y_test = train_val_test_sets(csv_path, test_size)

    train_data = Generator(X_train_new, Y_train_new)
    eval_data = Generator(X_eval, Y_eval)
    test_data = Generator(X_test, Y_test)

    train_sampler = RandomSampler(train_data)
    dev_sampler = RandomSampler(eval_data)
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)
    eval_dataloader = DataLoader(eval_data, sampler=dev_sampler, batch_size=16)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)

    # print(len(train_dataloader), len(eval_dataloader), len(test_dataloader))

    return train_dataloader, eval_dataloader, test_dataloader
