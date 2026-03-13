import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

from translate import translate_classes

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]), # using imagenet stats
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def load_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Train only the head
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_loss(predictions, gt):
    return F.cross_entropy(predictions, gt).mean()

def train_step(model, loader, optimizer, device):
    model.train()
    train_loss = []

    for images, labels in loader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = compute_loss(predictions, labels)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.cpu().data.numpy())

    return np.mean(train_loss)


def evaluate(model, loader, device):
    model.eval()
    accuracy = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            predictions = logits.argmax(dim=1)
            accuracy.append((predictions == labels).float().mean().item())

    return np.mean(accuracy)


def load_dataset(path, train_transform, val_transform):
    dataset = ImageFolder(path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = get_transforms()

    train_dataset, val_dataset = load_dataset(args.data_dir, train_transform, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    num_classes = len(train_dataset.dataset.classes)

    model = load_model(num_classes).to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    print("Training begins!")
    best_val_acc = 0.
    for epoch in range(args.epochs):
        train_loss = train_step(
            model, train_loader, optimizer, device
        )

        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{args.epochs}", 
            f"train_loss={train_loss:.4f}", 
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            
            classes = translate_classes(train_dataset.dataset.classes)
            
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes
            }, os.path.join(args.output_dir, "image_model.pt"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 image classifier"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/img_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
