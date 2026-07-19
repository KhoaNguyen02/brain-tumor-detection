import pandas as pd
import torch
from tqdm import tqdm


def _dice_score(pred_mask, target_mask, eps=1e-6):
    pred_flat = pred_mask.flatten(1)
    target_flat = target_mask.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def train_classifier(model, train_loader, val_loader, test_loader, device, criterion, optimizer,
                    n_epochs=100, scheduler=None, early_stopping=None):
    history = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

        train_loss /= len(train_loader.dataset)

        val_loss, val_acc = _evaluate_classifier(model, val_loader, device, criterion)
        history.append([train_loss, val_loss, val_acc])

        tqdm.write(f'loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        if early_stopping:
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                tqdm.write(
                    f'Early stopping at epoch {epoch + 1} (best epoch {early_stopping.best_epoch + 1})')
                early_stopping.load_best_model(model)
                break

        if scheduler is not None:
            scheduler.step(val_loss)

    history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'val_acc'])
    test_loss, test_acc = _evaluate_classifier(model, test_loader, device, criterion)
    return model, history, {'loss': test_loss, 'accuracy': test_acc}


def _evaluate_classifier(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_segmentation(model, train_loader, val_loader, test_loader, device, criterion, optimizer,
                    n_epochs=100, scheduler=None, early_stopping=None, seg_threshold=0.5):
    history = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

        train_loss /= len(train_loader.dataset)

        val_loss, val_dice = _evaluate_segmentation(model, val_loader, device, criterion, seg_threshold)
        history.append([train_loss, val_loss, val_dice])

        tqdm.write(f'loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_dice: {val_dice:.4f}')

        if early_stopping:
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                tqdm.write(
                    f'Early stopping at epoch {epoch + 1} (best epoch {early_stopping.best_epoch + 1})')
                early_stopping.load_best_model(model)
                break

        if scheduler is not None:
            scheduler.step(val_loss)

    history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'val_dice'])
    test_loss, test_dice = _evaluate_segmentation(model, test_loader, device, criterion, seg_threshold)
    return model, history, {'loss': test_loss, 'dice': test_dice}


def _evaluate_segmentation(model, loader, device, criterion, seg_threshold=0.5):
    model.eval()
    total_loss, dice_scores = 0.0, []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)

            pred_mask = (torch.sigmoid(logits.squeeze(1)) > seg_threshold).float()
            dice_scores.append(_dice_score(pred_mask, masks))

    avg_loss = total_loss / len(loader.dataset)
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return avg_loss, avg_dice


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.epochs_no_improve = 0
            self.best_epoch = epoch

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)
