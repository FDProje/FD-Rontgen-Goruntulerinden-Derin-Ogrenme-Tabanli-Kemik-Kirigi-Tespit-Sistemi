import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import label_binarize

# ----------- HAZIRLIK -----------
base_dir = r'C:\Users\asyao\PycharmProjects\MICROFRACTURES\fdataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = r'/content/drive/MyDrive/fdataset/fdataset/train/overlay'
val_path = r'/content/drive/MyDrive/fdataset/fdataset/val/overlay'
test_path = r'/content/drive/MyDrive/fdataset/fdataset/test/overlay'
num_classes = 7
image_size = (224, 224)


# ----------- VERI ÖNİŞLEME -----------
class NormalImageDataset(Dataset):
    def __init__(self, normal_dir, transform=None):
        self.normal_dir = normal_dir
        self.transform = transform
        self.normal_images = []
        self.labels = []
        for label_folder in sorted(os.listdir(self.normal_dir)):
            label_folder_normal = os.path.join(self.normal_dir, label_folder)
            if os.path.isdir(label_folder_normal):
                image_files = sorted(os.listdir(label_folder_normal))
                for image_file in image_files:
                    normal_img = os.path.join(label_folder_normal, image_file)
                    self.normal_images.append(normal_img)
                    self.labels.append(int(label_folder.split('_')[1]))

    def __getitem__(self, idx):
        self.label = self.labels[idx] - 1
        normal_img = Image.open(self.normal_images[idx]).convert("RGB")
        if self.transform:
            normal_img = self.transform(normal_img)
        return normal_img, self.label

    def __len__(self):
        return len(self.normal_images)


# ----------- MODEL MİMARİSİ -----------
import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models


class EfficientNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(EfficientNet, self).__init__()

        self.backbone = models.efficientnet_b7(weights="IMAGENET1K_V1" if pretrained else None)

        in_features = self.backbone.classifier[1].in_features 
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes) 

    def forward(self, x):
        return self.backbone(x)

# ----------- TRANSFORMERLAR VE VERİ YÜKLEME -----------
def dataset(batch_size=32):
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=15, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224)),
    ])
    train_dataset = NormalImageDataset(normal_dir=train_path, transform=transform)
    val_dataset = NormalImageDataset(normal_dir=val_path, transform=normalize_transform)
    test_dataset = NormalImageDataset(normal_dir=test_path, transform=normalize_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ----------- EĞİTİM VE DEĞERLENDİRME -----------
def save_checkpoint(model, optimizer, epoch, val_loss, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from epoch {epoch + 1}")
    return model, optimizer, epoch, val_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # Predict class labels
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print("Epoch {} | Loss: {:.4f} | Accuracy: {:.4f}".format(epoch + 1, running_loss / len(train_loader), correct / total))

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss)

    return model, history


# ----------- EĞİTİM SONU DEĞERLENDİRME -----------
def plot_metrics(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.show()


# ----------- TEST -----------
def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds, num_classes)
    plot_roc_curve(all_labels, all_preds, num_classes)


def plot_confusion_matrix(all_labels, all_preds, num_classes):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(1, num_classes + 1)],
                yticklabels=[str(i) for i in range(1, num_classes + 1)], cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(all_labels, all_preds, num_classes):
    y_true = label_binarize(all_labels, classes=[i for i in range(1, num_classes + 1)])
    fpr, tpr, _ = roc_curve(y_true.ravel(), np.array(all_preds).ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def main():
    print("Setup")
    num_classes = 7
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    train_loader, val_loader, test_loader = dataset(batch_size=batch_size)
    model = EfficientNet(pretrained=True, num_classes=num_classes).to(device)

    print("Model Optizing")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print(f"Criterion: ", criterion,"Optimizer: ", optimizer, "Scheduler: ", scheduler)

    print("Training")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    model.save('model_26.pt')

    print("Plotting")
    plot_metrics(history)

    print("Testing")
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
