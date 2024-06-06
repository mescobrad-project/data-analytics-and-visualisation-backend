import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datetime import datetime
import argparse
import torch
import matplotlib.pyplot as plt

from app.utils.conv3D import Conv3D
#from app.utils.resnet18_3d import ResNet18_3D
from app.utils.mri_dataloaders import train_eval_dataloaders
from app.utils.training import train_eval_model
#from app.utils.testing import test_on_multiple_mris

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

def run_experiment(data_path,
                   csv_path,
                   iterations,
                   batch_size,
                   lr,
                   early_stopping_patience
                   ):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = NeurodesktopStorageLocation + f'/model_data/saved_models_{timestamp}/'
    os.makedirs(exp_dir)

    # hyperparams
    #batch_size = 5
    #lr = 0.001
    scheduler_step_size = 10  #StepLR
    scheduler_gamma = 0.75   #StepLR

    for i in range(iterations):
        print(" ----- Currently on iteration no. {} ----- ".format(i+1), flush=True)
        
        train_dataloader, eval_dataloader = train_eval_dataloaders(data_path,
                                                                   csv_path,
                                                                   batch_size)

        model = Conv3D()

        #training
        train_losses_per_epoch, val_losses_per_epoch, train_accs, \
            dev_accs, train_f1s, dev_f1s, best_model, es_epoch = train_eval_model(train_dataloader,
                                                                                  eval_dataloader,
                                                                                  model,
                                                                                  lr,
                                                                                  scheduler_step_size,
                                                                                  scheduler_gamma,
                                                                                  early_stopping_patience)

        torch.save(trained_model.state_dict(), '../saved_models/' + f'{type(model).__name__}_experiment{i+1}.pth')

        # Plotting train and validation metrics
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        # Loss plot
        axs[0].plot(train_losses_per_epoch, label='Train Loss')
        axs[0].plot(val_losses_per_epoch, label='Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Train and Validation Loss per Epoch \n Early Stopping checkpoint at epoch {es_epoch}')
        axs[0].legend()
        axs[0].grid(True)

        # Accuracy plot
        axs[1].plot(train_accs, label='Train Accuracy')
        axs[1].plot(dev_accs, label='Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Train and Validation Accuracy per Epoch')
        axs[1].legend()
        axs[1].grid(True)

        # F1 Score plot
        axs[2].plot(train_f1s, label='Train F1 Score')
        axs[2].plot(dev_f1s, label='Validation F1 Score')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('F1 Score')
        axs[2].set_title('Train and Validation F1 Score per Epoch')
        axs[2].legend()
        axs[2].grid(True)

        # Save and show the plot
        plt.tight_layout()
        plt.savefig(exp_dir + f'train_val_metrics_plot_experiment{i + 1}.png')
        plt.show()

        # Save hyperparams to a text file
        with open(exp_dir + f'hyperparams_experiment{i + 1}.txt','w') as f:
            f.write(f'type: {type}\n')
            f.write(f'batch_size: {batch_size}\n')
            #f.write(f'eval_size: {eval_size}\n')
            f.write(f'lr: {lr}\n')
            f.write(f'early_stopping_patience: {early_stopping_patience}\n')

    return True
