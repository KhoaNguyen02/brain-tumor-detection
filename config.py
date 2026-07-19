import random

import albumentations as A
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from preprocessing import CLASS_NAMES
from utils import EarlyStopping

seed = 1234


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

set_seed(seed)

######################################################################################


class SegmentationConfig:
    def __init__(self, model, **kwargs):
        self.model = model
        self.model_name = 'Segmenter'
        self.seed = kwargs.get('seed', seed)

        self.image_size = kwargs.get('image_size', 256)

        self.batch_size = kwargs.get('batch_size', 8)
        self.lr = kwargs.get('lr', 1e-3)
        self.lr_factor = kwargs.get('lr_factor', 0.5)
        self.lr_patience = kwargs.get('lr_patience', 5)
        self.w_decay = kwargs.get('w_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 100)

        self.seg_threshold = kwargs.get('seg_threshold', 0.5)
        self.min_tumor_area_fraction = kwargs.get('min_tumor_area_fraction', 0.001)

        self.train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
        self.test_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=1e-6)
        self.early_stopping = EarlyStopping(patience=kwargs.get('patience', 10), min_delta=kwargs.get('min_delta', 0.001))


class ClassifierConfig:
    def __init__(self, model, **kwargs):
        self.model = model
        self.model_name = 'Classifier'
        self.seed = kwargs.get('seed', seed)

        self.image_size = kwargs.get('image_size', 224)
        self.num_classes = kwargs.get('num_classes', len(CLASS_NAMES))

        self.batch_size = kwargs.get('batch_size', 32)
        self.lr = kwargs.get('lr', 1e-3)
        self.lr_factor = kwargs.get('lr_factor', 0.5)
        self.lr_patience = kwargs.get('lr_patience', 5)
        self.w_decay = kwargs.get('w_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 100)

        self.train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
            A.GaussNoise(p=0.3), A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
        self.test_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=1e-6)
        self.early_stopping = EarlyStopping(patience=kwargs.get('patience', 10), min_delta=kwargs.get('min_delta', 0.001))
