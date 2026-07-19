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

    return max_class, max_confidence, class_probabilities