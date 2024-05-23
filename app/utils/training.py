import torch
import numpy as np
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(train_dataloader, model, optimizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device: ", device, flush=True)
    model.to(device)
    model.train()
    
    train_losses = []

    for step, batch in enumerate(train_dataloader):
        #batch = tuple(t.to(device) for t in batch)
        mri, labels_binary = batch
        mri, labels_binary = mri.to(device), labels_binary.to(device)
        optimizer.zero_grad()
        outputs = model(x=mri, labels=labels_binary)
        loss = outputs[0]
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        torch.cuda.empty_cache()  #Clear cache after each training step

    return np.average(train_losses)

def evaluate_model(eval_dataloader, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    model.to(device)
    model.eval()
    
    valid_losses = []
    eval_predictions = []
    eval_targets = []

    for batch in eval_dataloader:
        #batch = tuple(t.to(device) for t in batch)
        mri, labels_binary = batch
        mri, labels_binary = mri.to(device), labels_binary.to(device)
        with torch.no_grad():
            outputs = model(x=mri, labels=labels_binary)
            loss, logits = outputs[0], outputs[1]
        valid_losses.append(loss.item())

        logits = logits.detach().cpu().numpy()
        labels = labels_binary.to('cpu').numpy()

        logits = np.argmax(logits, axis=1)
        eval_predictions = np.append(eval_predictions, logits)
        eval_targets = np.append(eval_targets, labels)

        torch.cuda.empty_cache()  # Clear cache after each evaluation step

    valid_loss = np.average(valid_losses)

    return valid_loss, eval_targets, eval_predictions

def train_eval_model(train_dataloader,
                     eval_dataloader,
                     model,
                     lr,
                     es_patience,
                     scheduler_step_size,
                     scheduler_gamma):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # for early stopping
    threshold = 10e+5
    patience_counter = 0
    best_model = None
    
    n_epochs = 1000

    # Lists to store losses for plotting
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    for epoch in range(n_epochs):

        # Training phase
        train_loss = train_model(train_dataloader, model, optimizer)
        train_losses_per_epoch.append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Validation phase
        valid_loss, eval_targets, eval_predictions = evaluate_model(eval_dataloader, model)
        val_losses_per_epoch.append(valid_loss)
        scheduler.step()#scheduler.step(valid_loss)
        dev_f1 = metrics.f1_score(eval_targets, eval_predictions, zero_division=1)
        dev_acc = metrics.accuracy_score(eval_targets, eval_predictions)

        print(f'epoch {epoch+1} - train loss {train_loss:.3f} - val loss {valid_loss:.3f} - val f1 {dev_f1:.3f} - val acc {dev_acc:.3f} - lr {current_lr:.5f}', flush=True)

        # Early Stopping
        if round(valid_loss, 3) < threshold: #improvement
            patience_counter = 0
            best_model = model
            threshold = round(valid_loss, 3)
        else:
            patience_counter += 1
        
        if patience_counter == es_patience:
            es_epoch = max(epoch + 1 - es_patience, 1)
            print(f'Early Stopping checkpoint at epoch {es_epoch}. Patience value was {es_patience}.', flush=True)
            print('Train-Eval stage complete!', flush=True)

            '''
            for epoch in range(es_epoch):
                # Train the best_model on the validation data as well
                optimizer = torch.optim.Adam(best_model.parameters(), lr=lr) # leave lr value as the original one?
                train_model(eval_dataloader, best_model, optimizer)
            print('Training on eval data complete!', flush=True)
            '''
                
            break

    return train_losses_per_epoch, val_losses_per_epoch, best_model
