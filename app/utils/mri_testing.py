import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from app.utils.mri_dataloaders import test_dataloader
import matplotlib.pyplot as plt
import seaborn as sns

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

'''
def mri_prediction(model_path,
                   mri_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    nparray = nib.load(mri_path).get_fdata()
    tarray = torch.from_numpy(nparray).unsqueeze(0).unsqueeze(0).to(device)  # tarray.shape is torch.Size([1, 1, 157, 256, 256])

    logits = model(tarray)[1]
    #label = int(torch.argmax(logits))
    probs = F.softmax(logits, dim=1)
    top_prob, top_class = torch.max(probs, dim=1)
    if top_class == 0:
        group = 'Epilepsy (fcd)'
    elif top_class == 1:
        group = 'Non-Epilepsy (hc)'
    print(f'Model prediction: {group} with probability {round(top_prob[0].item(),2)}')

    # with open(output_path + f'prediction_for_{mri_path}.txt', 'w') as f:
    #    f.write(f'The predicted class for the test point located at {mri_path} is {label}\n')

    return True
'''

def mris_batch_prediction(model_path,
                          data_path,
                          csv_path,
                          output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    dataloader = test_dataloader(data_path,
                                 csv_path)

    test_predictions = []
    test_targets = []

    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from dataloader
        mri, labels_binary = batch

        with torch.no_grad():

            outputs = model(x = mri, labels = labels_binary)
            logits = outputs[1] #model raw output

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        labels = labels_binary.to('cpu').numpy()

        test_predictions = np.append(test_predictions, np.argmax(logits, axis=1))
        test_targets = np.append(test_targets, labels)

    class_names = ['fcd', 'hc']

    # Classification report
    report = classification_report(test_targets, test_predictions, target_names=class_names)
    fig, ax = plt.subplots()  # Adjust the figure size as needed
    ax.axis('off')
    fig.suptitle('Classification Report', y=0.8, x=0.5, fontweight='bold')
    # ax.set_title('Classification report')
    plt.text(0.01, 0.5, report, {'fontsize': 12}, fontproperties='monospace')
    plt.tight_layout()
    plt.subplots_adjust(top=0.5)
    plt.savefig(os.path.join(output_path, 'classification_report.png'))
    plt.show()

    # Confusion matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(test_targets, test_predictions)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax, cbar=False, xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    fig.suptitle('Confusion Matrix', y=0.7, x=0.5, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.55)
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    plt.show()

    return True




# path = NeurodesktopStorageLocation + "/model_data/saved_models_2024-06-12_17-47/"
# model_path = path + "conv3d_experiment1.pth"
# data_path = path + "mris_test/"
# csv_path = path + "groups_test.tsv"
# output_path = path
# mris_batch_prediction(model_path,
#                       data_path,
#                       csv_path,
#                       output_path)
