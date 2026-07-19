import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


BRISC_ROOT = 'brisc2025'
CLASS_NAMES = ['healthy', 'glioma', 'meningioma', 'pituitary']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TUMOR_CODE_TO_CLASS = {'gl': 'glioma', 'me': 'meningioma', 'pi': 'pituitary'}


def _get_tumor_code(filename):
    return os.path.splitext(filename)[0].split('_')[3]


def register_clf_data(train_root, test_root):
    records = []
    for split, root in [('train', train_root), ('test', test_root)]:
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.jpg'):
                    records.append({
                        'image_path': os.path.join(cls_dir, fname), 'filename': fname,
                        'class_name': cls, 'label': CLASS_TO_IDX[cls], 'split': split,
                    })

    return records


def register_seg_data(train_root, test_root):
    records = []
    for split, root in [('train', train_root), ('test', test_root)]:
        images_dir = os.path.join(root, 'images')
        masks_dir = os.path.join(root, 'masks')
        for fname in sorted(os.listdir(images_dir)):
            if fname.endswith('.jpg'):
                mask_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + '.png')
                class_name = TUMOR_CODE_TO_CLASS[_get_tumor_code(fname)]
                records.append({
                    'image_path': os.path.join(images_dir, fname), 'mask_path': mask_path,
                    'filename': fname, 'class_name': class_name, 'label': CLASS_TO_IDX[class_name],
                    'split': split,
                })

    return records


def load_classification_data(root=BRISC_ROOT, val_ratio=0.1, seed=None):
    assert os.path.exists(root), (
        f"BRISC2025 dataset not found at '{root}'. Make sure the dataset folder is "
        f"present in the project directory.")

    registry = register_clf_data(
        os.path.join(root, 'classification_task', 'train'),
        os.path.join(root, 'classification_task', 'test')
    )

    train_samples = [r for r in registry if r['split'] == 'train']
    test_samples = [r for r in registry if r['split'] == 'test']

    labels = [s['label'] for s in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=val_ratio, random_state=seed, stratify=labels)

    return train_samples, val_samples, test_samples


def load_segmentation_data(root=BRISC_ROOT, val_ratio=0.1, seed=None):
    assert os.path.exists(root), (
        f"BRISC2025 dataset not found at '{root}'. Make sure the dataset folder is "
        f"present in the project directory.")

    registry = register_seg_data(
        os.path.join(root, 'segmentation_task', 'train'),
        os.path.join(root, 'segmentation_task', 'test')
    )

    train_samples = [r for r in registry if r['split'] == 'train']
    test_samples = [r for r in registry if r['split'] == 'test']

    labels = [s['label'] for s in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=val_ratio, random_state=seed, stratify=labels)

    return train_samples, val_samples, test_samples


class ClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(sample['label'], dtype=torch.long)


class SegmentationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.float()
