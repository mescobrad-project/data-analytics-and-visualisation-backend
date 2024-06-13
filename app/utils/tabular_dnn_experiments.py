import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils.tabular_dnn import DenseNN
from app.utils.tabular_dnn_preprocessing import dataloaders
from app.utils.training import train_eval_model

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

def tabular_run_experiment(csv_path,
                           no_of_features,
                           test_size,
                           iterations,
                           lr,                           early_stopping_patience
                           ):

    '''

    Outputs
    - best model wrt early stopping criterion - pth file
    - png image saved at the model path - losses vs epochs and f1s vs epochs

    '''

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = NeurodesktopStorageLocation + f'/model_data/saved_tabular_dnn_models_{timestamp}/'
    os.makedirs(exp_dir)

    # hyperparams
    scheduler_step_size = 1  # no StepLR
    scheduler_gamma = 1  # no StepLR

    for i in range(iterations):
        print(" ----- Currently on iteration no. {} ----- ".format(i + 1), flush=True)

        train_dataloader, eval_dataloader, test_dataloader = dataloaders( NeurodesktopStorageLocation  + "/mescobrad_dataset.csv", test_size)

        model = DenseNN(input_size=no_of_features)

        # --- TRAIN / VALIDATION ---
        train_losses_per_epoch, val_losses_per_epoch, _, _, \
            train_f1s, dev_f1s, best_model, es_epoch = train_eval_model(train_dataloader,
                                                                        eval_dataloader,
                                                                        model,
                                                                        lr,
                                                                        scheduler_step_size,
                                                                        scheduler_gamma,
                                                                        early_stopping_patience)

        torch.save(best_model, exp_dir + f'tabular_dnn_experiment{i + 1}.pth')

        # Plotting train and validation metrics
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))

        # Losses plot
        axs[0].plot(train_losses_per_epoch, label='Train Loss')
        axs[0].plot(val_losses_per_epoch, label='Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Train and Validation Loss per Epoch \n Early Stopping checkpoint at epoch {es_epoch}')
        axs[0].legend()
        axs[0].grid(True)

        # f1s plot
        axs[1].plot(train_f1s, label='Train F1')
        axs[1].plot(dev_f1s, label='Validation F1')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('f1 score')
        axs[1].set_title('Train and Validation f1 score per Epoch')
        axs[1].legend()
        axs[1].grid(True)

        # Save and show the plot
        plt.tight_layout()
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

        cm = confusion_matrix(test_targets, test_predictions)
        print(cm)
        # class_names = ['class0', 'class1']
        classification_report_text = classification_report(test_targets,
                                                           test_predictions)  # , target_names=class_names)

        fig, ax = plt.subplots(2, 1, figsize=(5, 7))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('Actual')
        ax[0].set_title('Confusion Matrix')

        # Plot the classification report
        ax[1].axis('off')
        # ax[1].set_axis_off()
        combined_text = "Classification Report\n\n" + classification_report_text
        ax[1].text(0.01, 0.5, combined_text, fontsize=10, ha='left', va='center', transform=ax[1].transAxes,
                   family='monospace')

        #
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'test_performance_experiment{i + 1}.png'))
        plt.show()

    return True

# tabular_run_experiment("",
#                            12,
#                            0.1,
#                            1,
#                            0.001,
#                            2
#                            )
