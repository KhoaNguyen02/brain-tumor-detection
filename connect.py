from config import *
from training import *
from preprocessing import load_single_img


def get_model(model_name):
    if model_name == "CNN":
        _, model = load_model("CNN", get_model=True, device=device)
        config = CNNConfig(model)
    elif model_name == "DenseNet":
        _, model = load_model("DenseNet", get_model=True, device=device)
        config = DenseNetConfig(model)
    elif model_name == "ResNet":
        _, model = load_model("ResNet", get_model=True, device=device)
        config = ResNetConfig(model)
    elif model_name == "Auto":
        # Load all three models
        _, cnn = load_model("CNN", get_model=True, device=device)
        _, densenet = load_model("DenseNet", get_model=True, device=device)
        _, resnet = load_model("ResNet", get_model=True, device=device)
        models = {"CNN": cnn.eval(), "DenseNet": densenet.eval(), "ResNet": resnet.eval()}
        configs = {"CNN": CNNConfig(cnn), "DenseNet": DenseNetConfig(
            densenet), "ResNet": ResNetConfig(resnet)}
        return models, configs
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    model.eval()
    return model, config


def process_image(uploaded_file):
    with open("temp/uploaded_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp/uploaded_img.jpg"


def predict_image(model, config, image_path, device):
    if isinstance(model, dict):  # If using "Auto" model with multiple models
        # Define weights for each model
        model_weights = {
            "CNN": 0.35,    # Adjust these weights as needed
            "DenseNet": 0.35,
            "ResNet": 0.3
        }

        combined_probs = {}
        for model_name, m in model.items():
            image = load_single_img(image_path, transform=config[model_name].test_transform)
            _, _, class_probs = predict(m, image, device)

            for condition, prob in class_probs.items():
                if condition not in combined_probs:
                    combined_probs[condition] = 0
                combined_probs[condition] += model_weights[model_name] * prob

        # Normalize the weighted probabilities
        total_prob = sum(combined_probs.values())
        normalized_probs = {condition: (prob / total_prob)
                            for condition, prob in combined_probs.items()}

        # Determine the highest probability and corresponding prediction
        final_prediction = max(normalized_probs, key=normalized_probs.get)

        # Convert to percentage
        final_confidence = normalized_probs[final_prediction] * 100
        normalized_probs = {k: v * 100 for k, v in normalized_probs.items()}
        return final_prediction, final_confidence, normalized_probs
    else:
        # Single model scenario
        image = load_single_img(image_path, transform=config.test_transform)
        return predict(model, image, device)