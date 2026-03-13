import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_model(model_path):
    model_checkpoint = torch.load(model_path, map_location="cpu")

    classes = model_checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(model_checkpoint["state_dict"])

    return model, classes


def classify_image(image_path, model, classes):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    prediction = classes[predicted_idx.item()]

    return prediction, confidence.item()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify image with animal using ResNet-18"
    )

    parser.add_argument("--model_dir", type=str, default="models/img_model/image_model.pt")
    parser.add_argument("--image", type=str, required=True)

    return parser.parse_args()
        
def main():
    args = parse_args()

    model, classes = load_model(args.model_dir)
    label, confidence = classify_image(args.image, model, classes)

    print(f"Prediction : {label} (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
