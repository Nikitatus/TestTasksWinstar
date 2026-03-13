import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    """A simple feedforward neural network with hidden layers and ReLU activations"""

    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=10):
        super().__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class ConvNet(nn.Module):
    """A simple convolutional neural network for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, X):
        return self.model(X)