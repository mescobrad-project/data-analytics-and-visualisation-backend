import torch
import numpy as np
from sklearn import metrics

def train_model(train_dataloader, model, optimizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    train_losses = []
    train_targets = []
    train_predictions = []

    for step, batch in enumerate(train_dataloader):
        mri, labels_binary = batch
        mri, labels_binary = mri.to(device), labels_binary.to(device)
        optimizer.zero_grad()
        outputs = model(x=mri, labels=labels_binary)
        loss, logits = outputs[0], outputs[1]
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        logits = logits.detach().cpu().numpy()
        logits = np.argmax(logits, axis=1)
        train_predictions = np.append(train_predictions, logits)

        labels = labels_binary.to('cpu').numpy()
        train_targets = np.append(train_targets, labels)

        torch.cuda.empty_cache()  #Clear cache after each training step

    train_loss = np.average(train_losses)

    return train_loss, train_targets, train_predictions

def evaluate_model(eval_dataloader, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    model.to(device)
    model.eval()
    
    valid_losses = []
    eval_targets = []
    eval_predictions = []

    for batch in eval_dataloader:
        mri, labels_binary = batch
        mri, labels_binary = mri.to(device), labels_binary.to(device)
        with torch.no_grad():
            outputs = model(x=mri, labels=labels_binary)
            loss, logits = outputs[0], outputs[1]
        valid_losses.append(loss.item())

        logits = logits.detach().cpu().numpy()
        logits = np.argmax(logits, axis=1)
        eval_predictions = np.append(eval_predictions, logits)

        labels = labels_binary.to('cpu').numpy()
        eval_targets = np.append(eval_targets, labels)

        torch.cuda.empty_cache()  # Clear cache after each evaluation step

    valid_loss = np.average(valid_losses)

    return valid_loss, eval_targets, eval_predictions

def train_eval_model(train_dataloader,
                     eval_dataloader,
                     model,
                     lr,
                     scheduler_step_size,
                     scheduler_gamma,
                     early_stopping_patience):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # for early stopping
    threshold = 10e+5
    patience_counter = 0
    best_model = None
    
    n_epochs = 1000

    # Lists to store losses for plotting
    train_losses_per_epoch = []; val_losses_per_epoch = []
    train_accs = []; dev_accs = []
    train_f1s = []; dev_f1s = []

    for epoch in range(n_epochs):

        # Training phase
        train_loss, train_targets, train_predictions = train_model(train_dataloader, model, optimizer)
        train_losses_per_epoch.append(train_loss)
        train_f1 = metrics.f1_score(train_targets, train_predictions, zero_division=1)
        train_f1s.append(train_f1)
        train_acc = metrics.accuracy_score(train_targets, train_predictions)
        train_accs.append(train_acc)

        current_lr = optimizer.param_groups[0]['lr']

        # Validation phase
        valid_loss, eval_targets, eval_predictions = evaluate_model(eval_dataloader, model)
        val_losses_per_epoch.append(valid_loss)
        scheduler.step() #for StepLR
        dev_f1 = metrics.f1_score(eval_targets, eval_predictions, zero_division=1)
        dev_f1s.append(dev_f1)
        dev_acc = metrics.accuracy_score(eval_targets, eval_predictions)
        dev_accs.append(dev_acc)

        print(f'epoch {epoch+1} - train loss {train_loss:.3f} - val loss {valid_loss:.3f} - val f1 {dev_f1:.3f} - val acc {dev_acc:.3f} - lr {current_lr:.5f}', flush=True)

        # Early Stopping
        if round(valid_loss, 3) < threshold: #improvement
            patience_counter = 0
            best_model = model
            threshold = round(valid_loss, 3)
        else:
            patience_counter += 1
        
        if patience_counter == early_stopping_patience:
            es_epoch = max(epoch + 1 - early_stopping_patience, 1)
            print(f'Early Stopping checkpoint at epoch {es_epoch}. Patience value was {early_stopping_patience}.', flush=True)
            print('Train-Eval stage complete!', flush=True)
                
            break

    return train_losses_per_epoch, val_losses_per_epoch, train_accs, dev_accs, train_f1s, dev_f1s, best_model, es_epoch
