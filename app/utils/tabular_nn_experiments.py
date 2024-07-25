import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils.tabular_dnn import DenseNN
from app.utils.tabular_ae import AutoEncoderNN
from app.utils.tabular_nn_preprocessing import dataloaders
from app.utils.training import train_eval_model

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

def tabular_run_experiment(csv_path,
                           no_of_features,
                           test_size,
                           model_type,
                           iterations,
                           lr,
                           early_stopping_patience
                           ):

    '''
    Outputs
    - best model wrt early stopping criterion - pth file
    - training: png image saved at the model path - losses vs epochs and f1s vs epochs
    - testing: confusion matrix and classification_report as png files
    '''

    assert model_type in ['dense_neural_network', 'autoencoder']

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if model_type =='dense_neural_network':
        exp_dir = NeurodesktopStorageLocation + f'/model_data/saved_tabular_dnn_models_{timestamp}/'
    else:
        exp_dir = NeurodesktopStorageLocation + f'/model_data/saved_tabular_ae_models_{timestamp}/'
    os.makedirs(exp_dir)

    # hyperparams
    scheduler_step_size = 1  # no StepLR
    scheduler_gamma = 1  # no StepLR

    for i in range(iterations):
        print(" ----- Currently on iteration no. {} ----- ".format(i + 1), flush=True)

        train_dataloader, eval_dataloader, test_dataloader = dataloaders(csv_path, test_size)

        if model_type == 'dense_neural_network':
            model = DenseNN(input_size=no_of_features)
        else:
            model = AutoEncoderNN(input_size=no_of_features)

        # --- TRAIN / VALIDATION ---
        train_losses_per_epoch, val_losses_per_epoch, _, _, \
            train_f1s, dev_f1s, best_model, es_epoch = train_eval_model(train_dataloader,
                                                                        eval_dataloader,
                                                                        model,
                                                                        lr,
                                                                        scheduler_step_size,
                                                                        scheduler_gamma,
                                                                        early_stopping_patience)

        if model_type == 'dense_neural_network':
            torch.save(best_model, exp_dir + f'tabular_dnn_experiment{i + 1}.pth')
        else:
            torch.save(best_model, exp_dir + f'tabular_ae_experiment{i + 1}.pth')

        # Plotting train and validation metrics
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))

        # Losses plot
        axs[0].plot(train_losses_per_epoch, label='Train Loss')
        axs[0].plot(val_losses_per_epoch, label='Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Train and Validation Loss per Epoch \n Early Stopping checkpoint at epoch {es_epoch}')
        axs[0].legend()
        #axs[0].grid(True)

        # f1s plot
        axs[1].plot(train_f1s, label='Train F1')
        axs[1].plot(dev_f1s, label='Validation F1')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('f1 score')
        axs[1].set_title('Train and Validation f1 score per Epoch')
        axs[1].legend()
        #axs[1].grid(True)

        # Save and show the plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)

        plt.savefig(os.path.join(exp_dir, f'train_val_metrics_plot_experiment{i + 1}.png'))
        plt.show()

        # --- TESTING ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model.to(device)
        best_model.eval()

        test_predictions = []
        test_targets = []

        for batch in test_dataloader:
            point, labels_binary = batch
            point, labels_binary = point.to(device), labels_binary.to(device)
            with torch.no_grad():
                outputs = best_model(x=point, labels=labels_binary)
                logits = outputs[1]  # model raw output
            logits = logits.detach().cpu().numpy()
            labels = labels_binary.to('cpu').numpy()
            test_predictions = np.append(test_predictions, np.argmax(logits, axis=1))
            test_targets = np.append(test_targets, labels)

        # Classification report
        report = classification_report(test_targets, test_predictions)#, output_dict=True)  # , target_names=class_names)
        fig, ax = plt.subplots()
        ax.axis('off')
        fig.suptitle('Classification Report', y=0.8, x=0.5, fontweight='bold')
        #ax.set_title('Classification report')
        plt.text(0.01, 0.5, report, {'fontsize': 12}, fontproperties='monospace')  # use a monospaced font
        plt.tight_layout()
        plt.subplots_adjust(top=0.5)
        plt.savefig(os.path.join(exp_dir, f'classification_report_experiment{i + 1}.png'))
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(test_targets, test_predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        fig.suptitle('Confusion Matrix', y=0.7, x=0.5, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.55)
        plt.savefig(os.path.join(exp_dir, f'confusion_matrix_experiment{i + 1}.png'))
        plt.show()

    return True


# tabular_run_experiment(NeurodesktopStorageLocation + "/mescobrad_autoencoder_dataset.csv",
#                        800,
#                        0.1,
#                        'autoencoder',
#                        4,
#                        0.01,
#                        40
#                        )
