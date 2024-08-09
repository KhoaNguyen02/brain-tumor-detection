import torch
from torchvision import transforms
from models import DenseNet121
from preprocessing import load_single_img
from models.inference import predict

MODEL = "pretrained/model_densenet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = DenseNet121(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def process_image(uploaded_file, transform):
    with open("temp/uploaded_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = load_single_img("temp/uploaded_img.jpg", transform=transform)
    return image


def predict_image(model, image, device):
    return predict(model, image, device)