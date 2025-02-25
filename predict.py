import torch
from torchvision import models
import numpy as np
from PIL import Image
import argparse
import json


parser = argparse.ArgumentParser(description="Predict image classes")

parser.add_argument('image_path', type=str, help='Path to image')
parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
parser.add_argument('--topk', type=int, default=5, help='Return top K predictions')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping file for category names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# LoadCheckpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu"))
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu"))
    return model

model = load_checkpoint(args.checkpoint)
model.eval()


def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)


# Predict Class

def predict(image_path, model, topk=5):
    image = process_image(image_path).to(torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        output = model.forward(image)
    probs, indices = torch.exp(output).topk(topk)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i.item()] for i in indices[0]]
    return probs[0].tolist(), classes

probs, classes = predict(args.image_path, model, args.topk)


#  Results

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
flower_names = [cat_to_name[i] for i in classes]

for i in range(len(flower_names)):
    print(f"{flower_names[i]}: {probs[i]*100:.2f}%")
