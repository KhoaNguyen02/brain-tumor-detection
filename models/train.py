import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_model(model, train_loader, val_loader, test_loader, device, criterion, optimizer, 
                scheduler=None, n_epochs=10, max_epochs_stop=3):
    """Train a PyTorch model using the given data loaders.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    test_loader : torch.utils.data.DataLoader
        Test data loader
    device : torch.device
        Device to use for training (CPU or GPU)
    criterion : torch.nn.Module
        Loss function to use
    optimizer : torch.nn.Module
        Optimizer to use for training the model (SGD, Adam, etc.)
    scheduler : torch.nn.Module, optional
        Learning rate scheduler (used for learning rate decay), by default None
    n_epochs : int, optional
        Number of epochs to train the model, by default 10
    max_epochs_stop : int, optional
        Maximum number of epochs to wait for the validation loss to improve, by default 3

    Returns
    -------
    (torch.nn.Module, pd.DataFrame, float, int)
        Trained model, training history, test accuracy, best epoch
    """

    # Early stopping initialization
    epochs_no_improve = 0
    min_val_loss = np.Inf
    max_val_acc = 0
    history = []

    for epoch in range(n_epochs):
        train_loss, val_loss, train_acc, val_acc = 0.0, 0.0, 0.0, 0.0
        model.train()

        start_time = time.time()

        # Initialize tqdm progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

                _, prediction = torch.max(outputs.data, 1)
                correct = prediction.eq(labels.data.view_as(prediction))
                accuracy = torch.mean(correct.type(torch.FloatTensor))
                train_acc += accuracy.item() * images.size(0)

                # Update the progress bar
                pbar.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
                pbar.update()

        elapsed_time = time.time() - start_time

        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                correct = prediction.eq(labels.data.view_as(prediction))
                accuracy = torch.mean(correct.type(torch.FloatTensor))
                val_acc += accuracy.item() * images.size(0)

            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            val_acc = val_acc / len(val_loader.dataset)

            history.append([train_loss, val_loss, train_acc, val_acc])

            # Epoch summary
            tqdm.write(f'{len(train_loader)}/{len(train_loader)} [==============================] - {elapsed_time:.0f}s {1000*elapsed_time/len(train_loader):.0f}ms/step - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')

            # Perform early stopping
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
                max_val_acc = val_acc
                best_model = model
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == max_epochs_stop:
                    best_epoch = epoch - max_epochs_stop + 1
                    tqdm.write(f'\nEarly Stopping! Total epochs: {epoch + 1}. Best epoch: {best_epoch} with val loss: {min_val_loss:.2f} and val accuracy: {100 * max_val_acc:.2f}%')
                    model.load_state_dict(best_model.state_dict())
                    model.optimizer = optimizer
                    history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
                    test_acc = eval_model(model, test_loader, device)
                    return model, history, test_acc, best_epoch

        if scheduler is not None:
            scheduler.step(val_loss)

    model.optimizer = optimizer
    history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    test_acc = eval_model(model, test_loader, device)

    return model, history, test_acc, epoch

def eval_model(model, test_loader, device):
    """Evaluate a PyTorch model using the given data loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    test_loader : torch.utils.data.DataLoader
        Test data loader
    device : torch.device
        Device to use for evaluation (CPU or GPU)

    Returns
    -------
    float
        Test accuracy of the model
    """
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            correct = prediction.eq(labels.data.view_as(prediction))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            test_acc += accuracy.item() * images.size(0)
    test_acc = test_acc / len(test_loader.dataset)

    return test_acc