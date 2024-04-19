import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from app.utils.mri_generator import MRI_Generator

def train_eval_dataloaders(data_path,
                           dataset_train,
                           eval_size,
                           batch_size):

    #data_path to the folder where the sub-000id folders are in ex. './content/'
    #dataset_train should be a dataframe with columns 'participant_id' and 'label'
    
    participants = [x[:9] for x in os.listdir(data_path) if 'sub' in x] #constumized to the dataset

    training_participants = dataset_train[dataset_train['participant_id'].isin(participants)] #filtered rows

    #X_train_ = training_participants['participant_id'].values #ids
    #labels_binary_train = training_participants['label'].values #labels
    
    X_eval = training_participants['participant_id'].values[-eval_size:]
    y_eval_binary = training_participants['label'].values[-eval_size:]
    X_train = training_participants['participant_id'].values[:-eval_size]
    y_train_binary = training_participants['label'].values[:-eval_size]

    #del X_train_
    #del labels_binary_train

    #y_train_binary = torch.LongTensor(y_train_binary)
    #y_eval_binary = torch.LongTensor(y_eval_binary)

    train_data = MRI_Generator(X_train, torch.LongTensor(y_train_binary), data_path)
    eval_data = MRI_Generator(X_eval, torch.LongTensor(y_eval_binary), data_path)

    #train_sampler = RandomSampler(train_data)
    #dev_sampler = RandomSampler(eval_data)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size) #batch_size=1 old value
    eval_dataloader = DataLoader(eval_data, sampler=RandomSampler(eval_data), batch_size=batch_size)
    #print('points in the dataloaders ', len(train_dataloader.dataset), len(eval_dataloader.dataset))

    return train_dataloader, eval_dataloader

def test_dataloader(data_path,
                    dataset_test,
                    batch_size):
    
    participants = [x[:9] for x in os.listdir(data_path) if 'sub' in x] #constumized to the dataset
    test_participants = dataset_test[dataset_test['participant_id'].isin(participants)] #filtered rows
    X_test = test_participants['participant_id'].values
    labels_binary_test = test_participants['label'].values
    #y_test_binary = torch.LongTensor(labels_binary_test)
    test_data = MRI_Generator(X_test, torch.LongTensor(labels_binary_test), data_path)
    #test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=RandomSampler(test_data), batch_size=batch_size)
    
    return test_dataloader
