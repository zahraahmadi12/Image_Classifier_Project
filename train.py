import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import json

parser = argparse.ArgumentParser(description="Train a deep learning model")

parser.add_argument('--data_dir', type=str, default='flowers', help='Path to dataset')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Checkpoint save path')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()


device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=64, shuffle=False)
}

# Load Pretrained Model & Define Classifier

model = models.vgg16(weights='IMAGENET1K_V1')
for param in model.features.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(4096, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier
model.to(device)


# Define Loss Function & Optimizer

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


# Training the Model

epochs = args.epochs
steps = 0
print_every = 5

print(" Training the model...")

for epoch in range(epochs):
    running_loss = 0
    model.train()

    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

            running_loss = 0
            model.train()


#Save Model

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'architecture': 'vgg16',
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs
}

torch.save(checkpoint, args.save_dir)
print(f" Model saved to {args.save_dir}")
