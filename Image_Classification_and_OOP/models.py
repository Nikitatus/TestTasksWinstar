import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface
from utils import flatten_images

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from helper_models import FeedForwardNet, ConvNet

class RandomForestModel(MnistClassifierInterface):
    """A simple wrapper around sklearn RandomForestClassifier"""

    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        X_flat = flatten_images(X)
        self.model.fit(X_flat, y)
        return self

    def predict(self, X):
        X_flat = flatten_images(X)
        return self.model.predict(X_flat)


class FeedForwardModel(MnistClassifierInterface):
    """Simple feed-forward MNIST classifier (PyTorch)"""

    def __init__(self, hidden_sizes=[128, 64], lr=1e-3, epochs=10, batch_size=64, num_classes=10):
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def train(self, X, y):
        X_flat = flatten_images(X)
        
        self.model = FeedForwardNet(
            input_size=X_flat.shape[-1],
            hidden_sizes=self.hidden_sizes
        )
        
        dataset = TensorDataset(
            torch.FloatTensor(X_flat),
            torch.LongTensor(y)
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                output = self.model(batch_X)
                loss = criterion(output, batch_y)

                loss.backward()

                optimizer.step()
    
        return self
    
    def predict(self, X):
        X_flat = flatten_images(X)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.FloatTensor(X_flat))
            return output.argmax(dim=1).numpy()


class CNNModel(MnistClassifierInterface):
    """CNN-based MNIST classifier (PyTorch)"""
    
    def __init__(self, lr=1e-3, epochs=10, batch_size=64, num_classes=10, device=torch.device('cpu')):
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _add_channel_dim(self, X):
        X = X[:, None, :, :]
        return X

    def train(self, X, y):
        X = self._add_channel_dim(X)

        self.model = ConvNet()
        self.model = self.model.to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                output = self.model(batch_X)
                loss = criterion(output, batch_y)

                loss.backward()

                optimizer.step()
        return self
    
    def predict(self, X):
        X = self._add_channel_dim(X)

        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            output = self.model(X)
            return output.argmax(dim=1).cpu().numpy()