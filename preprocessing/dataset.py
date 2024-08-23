import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_data(size=(512, 512), seed=None, download=False):
    """Load raw images and labels from dataset without preprocessing.

    Parameters
    ----------
    path : str, optional
        Path to dataset, by default 'dataset'

    Returns
    -------
    (list, list, list, list)
        Training images, training labels, testing images, testing labels
    """

    # Check if dataset is available
    path = 'dataset'

    assert download or os.path.exists(
        path), "Dataset not found. Set download=True to download the dataset."

    # If dataset is not available, download it
    if download:
        os.makedirs(path, exist_ok=True)
        snapshot_download(repo_id="Simezu/brain-tumour-MRI-scan", repo_type="dataset", allow_patterns="*.jpg",
                        local_dir=path, etag_timeout=60)

    # Get path of training and testing data
    train_path = os.path.join(path, 'Training')
    test_path = os.path.join(path, 'Testing')

    X, y = [], []

    # Read raw images and labels
    for label in os.listdir(train_path):
        for image in os.listdir(os.path.join(train_path, label)):
            if image.endswith('.jpg'):
                img = cv2.imread(os.path.join(train_path, label, image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                X.append(img)
                y.append(label)

    # Read raw testing images and labels
    for label in os.listdir(test_path):
        for image in os.listdir(os.path.join(test_path, label)):
            if image.endswith('.jpg'):
                img = cv2.imread(os.path.join(test_path, label, image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def crop_image(img, verbose=False):
    """Crop image based on extreme points.

    Parameters
    ----------
    img : np.ndarray
        Image to be cropped.
    verbose : bool, optional
        Plot the process of cropping if set to True, by default False.

    Returns
    -------
    np.ndarray
        Cropped image
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get threshold
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    max_countours = max(contours, key=cv2.contourArea)

    # Get extreme points
    left = tuple(max_countours[max_countours[:, :, 0].argmin()][0])
    right = tuple(max_countours[max_countours[:, :, 0].argmax()][0])
    top = tuple(max_countours[max_countours[:, :, 1].argmin()][0])
    bot = tuple(max_countours[max_countours[:, :, 1].argmax()][0])

    # Crop image based on extreme points
    cropped_image = img[top[1] + 2:bot[1] - 2, left[0] + 2:right[0] - 2]

    # Plot the process of cropping, if verbose is True
    if verbose:
        image_countour = cv2.drawContours(
            img.copy(), [max_countours], -1, (0, 255, 255), 4)
        points = cv2.circle(image_countour.copy(), left, 8, (0, 0, 255), -1)
        points = cv2.circle(points, right, 8, (0, 255, 0), -1)
        points = cv2.circle(points, top, 8, (255, 0, 0), -1)
        points = cv2.circle(points, bot, 8, (255, 255, 0), -1)

        _, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[1].imshow(image_countour)
        ax[1].set_title('Contour')
        ax[2].imshow(points)
        ax[2].set_title('Extreme Points')
        ax[3].imshow(cropped_image)
        ax[3].set_title('Cropped Image')
        plt.show()

    return cropped_image


def get_dataloader(train_data, val_data, test_data, batch_size=32, train_transform=None, test_transform=None, device='cpu'):
    """Get dataloader for training and testing.

    Parameters
    ----------
    X_train : np.ndarray
        Training images.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing images.
    y_test : np.ndarray
        Testing labels.
    batch_size : int, optional
        Batch size, by default 32
    transform : torchvision.transforms, optional
        Transformations to be applied to images, by default None
    device : str, optional
        Device to be used, by default 'cpu'

    Returns
    -------
    (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
        Training dataloader, validation dataloader, testing dataloader
    """

    train_dataset = BrainTumorDataset(
        *train_data, transform=train_transform, device=device)
    val_dataset = BrainTumorDataset(
        *val_data, transform=test_transform, device=device)
    test_dataset = BrainTumorDataset(
        *test_data, transform=test_transform, device=device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


class BrainTumorDataset(Dataset):
    """Dataset for brain tumor images."""

    def __init__(self, images, labels, transform=None, device='cpu'):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.device = device
        self.label_dict = {'1-notumor': 0, '2-glioma': 1, '3-meningioma': 2, '4-pituitary': 3}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = self.label_dict[label]
        image = crop_image(image)
        if self.transform:
            image = self.transform(image)

        return image, label


def load_single_img(path, transform):
    """Load a single image from path

    Parameters
    ----------
    path : str
        Path to image
    transform : torchvision.transforms
        Transformations to be applied to the image
    size : tuple, optional
        Size of the image, by default (256, 256)

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader containing the image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_image(img)
    img = transform(img)
    img = img.unsqueeze(0)

    new_data = DataLoader(img, batch_size=1, shuffle=False)
    return new_data