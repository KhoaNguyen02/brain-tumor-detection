import cv2
import numpy as np
import torch
import torch.nn.functional as F


def predict(model, image, device):
    """Get prediction and confidence level of a single image using a trained model.

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
    tuple
        Prediction for the image and confidence level in percentage
    """
    model.eval()
    with torch.no_grad():
        image = next(iter(image)).to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()

    pred_dict = {0: 'Normal', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'}

    # Create a dictionary of class probabilities
    class_probabilities = {pred_dict[i]: prob *100 for i, prob in enumerate(probabilities)}

    # Get the highest confidence prediction
    max_class = max(class_probabilities, key=class_probabilities.get)
    max_confidence = class_probabilities[max_class]

    return max_class, max_confidence

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
        pred = model(image, inference=True)
        cls = pred.argmax(dim=1)

        pred[:, cls].backward()
        gradients = model.get_gradient()
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