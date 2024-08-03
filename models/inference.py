import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def predict(model, image, device):
    """Get prediction of a single image using a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to use for prediction
    image : torch.Tensor
        The image tensor to be predicted
    device : torch.device
        Device to run the model on (CPU or GPU)

    Returns
    -------
    str
        Prediction for the image
    """
    model.eval()
    with torch.no_grad():
        image = next(iter(image)).to(device)
        output = model(image)
        _, prediction = torch.max(output.data, 1)
        prediction = prediction.item()

    pred_dict = {0: 'Normal', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'}
    return pred_dict[prediction]

def clear_hooks(layer):
    """Clears all hooks from a layer. Useful when reusing a model for inference.

    Parameters
    ----------
    layer : torch.nn.Module
        Layer to clear hooks from.
    """
    if hasattr(layer, '_backward_hooks'):
        layer._backward_hooks.clear()
    if hasattr(layer, '_full_backward_hooks'):
        layer._full_backward_hooks.clear()

def register_hooks(model):
    """Registers a full backward hook on the last convolutional layer of a model. Useful for Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        Model to register the hook on
    """
    def hook_function(module, grad_in, grad_out):
        model.gradients = grad_out[0]

    # find the last convolutional layer in the model
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Conv2d):
                    last_conv_layer = sub_layer

    if last_conv_layer is not None:
        # Register the full backward hook
        last_conv_layer.register_full_backward_hook(hook_function)
    else:
        raise ValueError("No Conv2d layer found in the model")

def generate_cam(model, input_data):
    """Generates class activation maps (CAM) for a given model and input data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to generate CAM for
    input_data : torch.Tensor or torch.utils.data.DataLoader
        Input data to generate CAM for (single image or DataLoader)

    Returns
    -------
    list of np.ndarray
        List of CAMs for the input data
    """

    def process_single_image(image):
        model.eval()
        pred = model(image)
        cls = pred.argmax(dim=1)

        pred[:, cls].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(image).detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()

        return heatmap

    if isinstance(input_data, torch.utils.data.DataLoader):
        heatmaps = []
        for images in input_data:
            for image in images:
                heatmap = process_single_image(image.unsqueeze(0))
                heatmaps.append(heatmap)
        return heatmaps
    else:
        return process_single_image(input_data)

def apply_heatmap(img, heatmap):
    """Applies a heatmap to an image. Useful for visualizing CAMs.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        Image to apply the heatmap to
    heatmap : np.ndarray
        Heatmap to apply to the image

    Returns
    -------
    np.ndarray
        Image with the heatmap applied
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    return superimposed_img