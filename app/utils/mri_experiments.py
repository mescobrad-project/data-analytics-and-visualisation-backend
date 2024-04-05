import os
from datetime import datetime
import argparse
import torch

from app.utils.participants import get_participants
from app.utils.conv3D import Conv3D_large, Conv3D_small
from app.utils.mri_dataloaders import train_eval_dataloaders
from app.utils.training import train_eval_model

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment with 3D Convolutional Neural Networks.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to run the experiment.")
    parser.add_argument("--participants_path", type=str, help="Path to participants data.")
    parser.add_argument("--data_path", type=str, help="Path to the main data directory.")
    parser.add_argument("--model_type", type=str, default="small", choices=["large", "small"], help="Type of Conv3D model to use (large or small).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--eval_size", type=int, default=8, help="No of participants for evaluation.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping during training.")
    return parser.parse_args()

def run_experiment(iterations, 
                   participants_path, 
                   data_path, 
                   model_type, 
                   batch_size, 
                   eval_size, 
                   lr, 
                   patience):
    
    assert model_type in ['large', 'small']
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(f'model_data/saved_models_{timestamp}/') #For value exist_ok=True leaves directory unaltered.
    
    for i in range(iterations):
        print(" ----- Currently on iteration no. {} ----- ".format(i+1))
        
        dataset_train, dataset_test = get_participants(participants_path)
        train_dataloader, eval_dataloader = train_eval_dataloaders(data_path, dataset_train, eval_size, batch_size)
        
        if model_type == 'large':
            model = Conv3D_large()
        else:
            model = Conv3D_small()
        
        trained_model = train_eval_model(train_dataloader, eval_dataloader, model, lr, patience)
        torch.save(trained_model, NeurodesktopStorageLocation + f'/model_data/saved_models_{timestamp}/' + f'{type(model).__name__}_experiment{i+1}.pth')
        #torch.save(trained_model.state_dict(), '../saved_models/' + f'{type(model).__name__}_experiment{i+1}.pth') 

        # θελουμε και testing εδω?        
    return True
        # print()

# if __name__ == "__main__":
#     args = parse_arguments()
#     run_experiment(args.iterations, args.participants_path, args.data_path, args.model_type, args.batch_size, args.eval_size, args.lr, args.patience)
