import numpy as np
import torch
from sklearn import metrics
import nibabel as nib
from app.utils.mri_dataloaders import test_dataloader


def mri_prediction(model_path, mri_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.eval()

    nparray = nib.load(mri_path).get_fdata()
    tarray = torch.from_numpy(nparray).unsqueeze(0).unsqueeze(0).to(device)
    #tarray.shape is torch.Size([1, 1, 157, 256, 256])

    pred = model(tarray)[1]

    return pred, torch.argmax(pred)


def test_on_multiple_mris(model_path, 
                          data_path, 
                          dataset_test, 
                          batch_size):
    
    '''
    - code that calculates test set statistics
    - data_path to the folder where the sub-000id folders are in ex. './content/'
    - dataset_test is dataframe with 'participant_id' and 'label' columns
    '''

    model = torch.load(model_path)

    test_predictions = []
    test_targets = []

    dataloader = test_dataloader(data_path,
                                dataset_test,
                                batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

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

    # test acc/precision/recall/f1 of binary dementia
    test_acc_d = metrics.accuracy_score(test_targets, test_predictions)
    test_precision_d = metrics.precision_score(test_targets, test_predictions, zero_division=1)
    test_recall_d = metrics.recall_score(test_targets, test_predictions, zero_division=1)
    test_specificity_d = metrics.recall_score(test_targets, test_predictions, pos_label = 0, zero_division=1)
    test_f1_d = metrics.f1_score(test_targets, test_predictions, zero_division=1)
    test_roc_auc_d = metrics.roc_auc_score(test_targets, test_predictions)

    return test_acc_d, test_precision_d, test_recall_d, test_specificity_d, test_f1_d, test_roc_auc_d
