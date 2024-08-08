from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (16, 112, 112)

            # Second Convolutional Block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (32, 56, 56)

            # Third Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (64, 28, 28)

            # Fourth Convolutional Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (128, 14, 14)

            # Fifth Convolutional Block (Optional for larger networks)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (256, 7, 7)
        )

        self.to_logits = nn.Sequential(
            # Flatten layer
            nn.Flatten(),  # output: 256 * 7 * 7 = 12544

            # Fully Connected Layers
            nn.Linear(256 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, num_classes),
        )

        self.gradient = None

    def forward(self, x, inference=False):
        output = self.layers(x)
        if inference:
            output.register_hook(self.activations_hook)
        output = self.to_logits(output)
        return output
    
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.layers(x)