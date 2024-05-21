import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import nibabel as nib
from app.utils.mri_dataloaders import test_dataloader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

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


def mris_batch_prediction(model_path,
                          data_path,
                          csv_path,
                          output_path,
                          batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    dataloader = test_dataloader(data_path,
                                 csv_path,
                                 batch_size)

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

    cm = confusion_matrix(test_targets, test_predictions)
    class_names = ['fcd', 'hc']
    #report = classification_report(test_targets, test_predictions, target_names=class_names, output_dict=True)
    classification_report_text = classification_report(test_targets, test_predictions, target_names=class_names)

    # Create a figure with two subplots: one for the heatmap and one for the text
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))#, gridspec_kw={'height_ratios': [3, 1]})

    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax[0], cbar=False)
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title('Confusion Matrix')

    # Plot the classification report
    ax[1].axis('off')
    #ax[1].set_title('Classification Report')
    combined_text = "Classification Report\n\n" + classification_report_text
    ax[1].text(0.01, 0.5, combined_text, fontsize=12, ha='left', va='center', transform=ax[1].transAxes, family='monospace')

    # Save the heatmap
    plt.savefig(output_path + 'test_performance.png')

    # Show the plot (optional, if you want to see the plot in addition to saving it)
    #plt.show()

    return True

    '''
    # test acc/precision/recall/f1 of binary dementia
    test_acc_d = metrics.accuracy_score(test_targets, test_predictions)
    test_precision_d = metrics.precision_score(test_targets, test_predictions, zero_division=1)
    test_recall_d = metrics.recall_score(test_targets, test_predictions, zero_division=1)
    test_specificity_d = metrics.recall_score(test_targets, test_predictions, pos_label = 0, zero_division=1)
    test_f1_d = metrics.f1_score(test_targets, test_predictions, zero_division=1)
    test_roc_auc_d = metrics.roc_auc_score(test_targets, test_predictions)

    return test_acc_d, test_precision_d, test_recall_d, test_specificity_d, test_f1_d, test_roc_auc_d
    '''
