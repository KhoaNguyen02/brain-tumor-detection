import random
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import *
from models import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


num_classes = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
set_seed(seed)

######################################################################################


class Config:
    def __init__(self, model, model_name, **kwargs):
        self.model = model
        self.model_name = model_name
        self.seed = kwargs.get('seed', seed)
        self.batch_size = kwargs.get('batch_size', 64)
        self.resize = kwargs.get('resize', (224, 224))
        self.lr = kwargs.get('lr', 1e-4)
        self.w_decay = kwargs.get('w_decay', 0.001)
        self.epochs = kwargs.get('epochs', 50)
        self.description = kwargs.get('description', '')

        # Data-related parameters
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(self.resize, antialias=True),
            transforms.ColorJitter(brightness=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(self.resize, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        self.scheduler = StepLR(self.optimizer, step_size=kwargs.get(
            'step_size', 20), gamma=kwargs.get('gamma', 0.1), verbose=True)
        self.early_stopping = EarlyStopping(patience=kwargs.get(
            'patience', 7), min_delta=kwargs.get('min_delta', 0.001))


class CNNConfig(Config):
    def __init__(self, model):
        super().__init__(
            model=model,
            model_name='CNN',
            lr=1e-4,
            description='Training a simple CNN model'
        )


class ResNetConfig(Config):
    def __init__(self, model):
        super().__init__(
            model=model,
            model_name='ResNet',
            lr=5e-4,
            description='Training a ResNet model'
        )


class DenseNetConfig(Config):
    def __init__(self, model):
        super().__init__(
            model=model,
            model_name='DenseNet',
            lr=1e-4,
            resize=(64, 64),
            gamma=0.1,
            description='Training a DenseNet model'
        )