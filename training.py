from config import *
from preprocessing import *
from models import *
import json

models = {
    "CNN": CNN(num_classes),
    "ResNet": ResNet101(num_classes),
    "DenseNet": DenseNet121(num_classes)
}


def save_model(model, history, test_acc, config):
    model_name = config.model_name
    early_stopping = config.early_stopping
    save_path = f'./pretrained/{model_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the model state dictionary
    torch.save(model.state_dict(), f'{save_path}/{model_name}.pth')

    # Convert DataFrame to a dictionary format for JSON serialization
    history_dict = history.to_dict(orient='list')

    # Prepare the data to save in JSON format
    save_data = {
        "history": history_dict,
        "test_accuracy": test_acc,
        "early_stopping": early_stopping.best_epoch
    }

    # Save the data to a JSON file
    with open(f'{save_path}/{model_name}_history.json', 'w') as json_file:
        json.dump(save_data, json_file, indent=4)


def load_model(model_name, get_model=False, device='cpu'):
    save_path = f'./pretrained/{model_name}'
    history_path = f'{save_path}/{model_name}_history.json'
    model_path = f'{save_path}/{model_name}.pth'

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No history found for model {model_name} at {history_path}")

    # Load history and other data
    with open(history_path, 'r') as json_file:
        data = json.load(json_file)

    if get_model:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        # Initialize the model
        model = models[model_name]
        # Load the model's state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return data, model
    else:
        return data


def train_cnn(save=False):
    model = models["CNN"].to(device)

    config = CNNConfig(model)
    train_data, val_data, test_data = load_data(seed=config.seed)
    train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, test_data, batch_size=config.batch_size,
                                                        train_transform=config.train_transform,
                                                        test_transform=config.test_transform,
                                                        device=device)
    model, history, test_acc = train_model(model, train_loader, val_loader, test_loader,
                                        device, config.criterion, config.optimizer,
                                        n_epochs=config.epochs, scheduler=config.scheduler, early_stopping=config.early_stopping)

    print("Training completed !!!")
    if save:
        save_model(model, history, test_acc, config)
        print(f"Model saved at ./pretrained/{config.model_name}")
    return model, history, test_acc


def train_resnet(save=False):
    model = models["ResNet"].to(device)

    config = ResNetConfig(model)
    train_data, val_data, test_data = load_data(seed=config.seed)
    train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, test_data, batch_size=config.batch_size,
                                                           train_transform=config.train_transform,
                                                           test_transform=config.test_transform,
                                                           device=device)
    model, history, test_acc = train_model(model, train_loader, val_loader, test_loader,
                                           device, config.criterion, config.optimizer,
                                           n_epochs=config.epochs, scheduler=config.scheduler, early_stopping=config.early_stopping)

    print("Training completed !!!")
    if save:
        save_model(model, history, test_acc, config)
        print(f"Model saved at ./pretrained/{config.model_name}")
    return model, history, test_acc


def train_densenet(save=False):
    model = models["DenseNet"].to(device)

    config = DenseNetConfig(model)
    train_data, val_data, test_data = load_data(seed=config.seed)
    train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, test_data, batch_size=config.batch_size,
                                                        train_transform=config.train_transform,
                                                        test_transform=config.test_transform,
                                                        device=device)
    model, history, test_acc = train_model(model, train_loader, val_loader, test_loader,
                                        device, config.criterion, config.optimizer,
                                        n_epochs=config.epochs, scheduler=config.scheduler, early_stopping=config.early_stopping)

    print("Training completed !!!")
    if save:
        save_model(model, history, test_acc, config)
        print(f"Model saved at ./pretrained/{config.model_name}")
    return model, history, test_acc