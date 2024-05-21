import os
from datetime import datetime
import argparse
import torch
import matplotlib.pyplot as plt

from app.utils.conv3D import Conv3D
from app.utils.mri_dataloaders import train_eval_dataloaders
from app.utils.training import train_eval_model
#from app.utils.testing import test_on_multiple_mris

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

def run_experiment(data_path,
                   csv_path,
                   iterations,
                   es_patience,
                   ):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = NeurodesktopStorageLocation + f'/model_data/saved_models_{timestamp}/'
    os.makedirs(exp_dir)

    # learning rate hyperparams
    lr = 0.001
    scheduler_step_size = 5
    scheduler_gamma = 0.75
    
    for i in range(iterations):
        print(" ----- Currently on iteration no. {} ----- ".format(i+1), flush=True)
        
        train_dataloader, eval_dataloader = train_eval_dataloaders(data_path, csv_path)

        model = Conv3D()

        #training
        train_losses_per_epoch, val_losses_per_epoch, trained_model = train_eval_model(train_dataloader,
                                                                                       eval_dataloader,
                                                                                       model,
                                                                                       lr,
                                                                                       es_patience,
                                                                                       scheduler_step_size,
                                                                                       scheduler_gamma)
        torch.save(trained_model, exp_dir + f'{type(model).__name__}_experiment{i+1}.pth')
        #torch.save(trained_model.state_dict(), '../saved_models/' + f'{type(model).__name__}_experiment{i+1}.pth')

        # Plotting train and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_per_epoch, label='Train Loss')
        plt.plot(val_losses_per_epoch, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(exp_dir + 'train_val_loss_plot_experiment{i+1}.png')
        plt.show()

        '''
        #testing
        acc, prec, rec, specif, f1, auc = test_on_multiple_mris(exp_dir + f'{type(model).__name__}_experiment{i+1}.pth',
                                                                data_path,
                                                                dataset_test,
                                                                1)
        # Save metrics to a text file
        with open(exp_dir + f'test_metrics_experiment{i + 1}.txt','w') as f:
            f.write(f'No. of test data: {len(dataset_test)}\n')
            f.write(f'Accuracy: {acc}\n')
            f.write(f'Precision: {prec}\n')
            f.write(f'Recall: {rec}\n')
            f.write(f'Specificity: {specif}\n')
            f.write(f'F1-Score: {f1}\n')
            f.write(f'AUC: {auc}\n')
        '''

        # Save hyperparams to a text file
        with open(exp_dir + f'hyperparams_experiment{i + 1}.txt','w') as f:
            #f.write(f'batch_size: {batch_size}\n')
            #f.write(f'eval_size: {eval_size}\n')
            f.write(f'lr: {lr}\n')
            f.write(f'es_patience: {es_patience}\n')
            f.write(f'scheduler_step_size: {scheduler_step_size}\n')
            f.write(f'scheduler_gamma: {scheduler_gamma}\n')

    return True
        # print()

# if __name__ == "__main__":
#     args = parse_arguments()
#     run_experiment(args.iterations, args.participants_path, args.data_path, args.model_type, args.batch_size, args.eval_size, args.lr, args.patience)
