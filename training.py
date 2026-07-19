import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from models import *
from preprocessing import *
from utils import train_classifier, train_segmentation


def save_model(model, history, test_metrics, config):
    save_path = f'./pretrained/{config.model_name}'
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), f'{save_path}/{config.model_name}.pth')

    save_data = {
        'history': history.to_dict(orient='list'),
        'test_metrics': test_metrics,
        'early_stopping_epoch': config.early_stopping.best_epoch,
    }
    with open(f'{save_path}/{config.model_name}_history.json', 'w') as f:
        json.dump(save_data, f, indent=4)


def load_model(model_name, model_class, get_model=False, device='cpu'):
    save_path = f'./pretrained/{model_name}'
    history_path = f'{save_path}/{model_name}_history.json'
    model_path = f'{save_path}/{model_name}.pth'

    if not os.path.exists(history_path):
        raise FileNotFoundError(f'No history found for model {model_name} at {history_path}')

    with open(history_path, 'r') as f:
        data = json.load(f)

    if get_model:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'No model found at {model_path}')
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return data, model

    return data


def train_classification_model(save=False, dataset_root='brisc2025', device='cpu'):
    model = ConvNext(num_classes=len(CLASS_NAMES)).to(device)
    config = ClassifierConfig(model)
    set_seed(config.seed)

    train_data, val_data, test_data = load_classification_data(root=dataset_root, seed=config.seed)

    train_loader = DataLoader(
        ClassificationDataset(train_data, transform=config.train_transform),
        batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(
        ClassificationDataset(val_data, transform=config.test_transform),
        batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(
        ClassificationDataset(test_data, transform=config.test_transform),
        batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    model, history, test_metrics = train_classifier(
        model, train_loader, val_loader, test_loader, device, criterion, config.optimizer,
        n_epochs=config.epochs, scheduler=config.scheduler, early_stopping=config.early_stopping)

    print(f"Training completed !!! Test accuracy: {test_metrics['accuracy']:.4f}")
    if save:
        save_model(model, history, test_metrics, config)
        print(f'Model saved at ./pretrained/{config.model_name}')

    return model, history, test_metrics


def train_segmentation_model(save=False, dataset_root='brisc2025', device='cpu'):
    model = AttentionUNet().to(device)
    config = SegmentationConfig(model)
    set_seed(config.seed)

    train_data, val_data, test_data = load_segmentation_data(root=dataset_root, seed=config.seed)

    train_loader = DataLoader(
        SegmentationDataset(train_data, transform=config.train_transform),
        batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(
        SegmentationDataset(val_data, transform=config.test_transform),
        batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(
        SegmentationDataset(test_data, transform=config.test_transform),
        batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    model, history, test_metrics = train_segmentation(
        model, train_loader, val_loader, test_loader, device, segmentation_loss, config.optimizer,
        n_epochs=config.epochs, scheduler=config.scheduler, early_stopping=config.early_stopping,
        seg_threshold=config.seg_threshold)

    print(f"Training completed !!! Test dice: {test_metrics['dice']:.4f}")
    if save:
        save_model(model, history, test_metrics, config)
        print(f'Model saved at ./pretrained/{config.model_name}')

    return model, history, test_metrics


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("#" * 20)
    print("Training Classification Model...")
    train_classification_model(save=True, device=device)

    print("#" * 20)
    print("Training Segmentation Model...")
    train_segmentation_model(save=True, device=device)
