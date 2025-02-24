import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Verisetlerinin Yolu
dir0='Eğitim Dataseti'  # Eğitim klasör yolu
dir2='Değerlendirme Dataseti'  # Değerlendirme klasör yolu

# Eğitim veriseti işleme
classes=[]
paths=[]
for dirname, _, filenames in os.walk(dir0):
    for filename in filenames:
        classes+=[dirname.split('/')[-1]]  # Sınıf isimlerini alıyoruz
        paths+=[(os.path.join(dirname, filename))]  # Dosya yollarını alıyoruz

# Değerlendirme veriseti işleme
tclasses=[]
tpaths=[]
for dirname, _, filenames in os.walk(dir2):
    for filename in filenames:
        tclasses+=[dirname.split('/')[-1]]  # Sınıf isimlerini alıyoruz
        tpaths+=[(os.path.join(dirname, filename))]  # Dosya yollarını alıyoruz

# Sınıfların etiketlerini oluşturuyoruz
N=list(range(len(classes)))
class_names=sorted(set(classes))  # Sınıf isimlerini sıralıyoruz
print(class_names)
normal_mapping=dict(zip(class_names,N))  # Sınıf isimlerini etiketlere eşliyoruz
reverse_mapping=dict(zip(N,class_names))  # Etiketleri sınıf isimleriyle eşliyoruz

# Eğitim verisetini DataFrame olarak düzenliyoruz
data=pd.DataFrame(columns=['path','class','label'])
data['path']=paths
data['class']=classes
data['label']=data['class'].map(normal_mapping)

# Değerlendirme verisetini DataFrame olarak düzenliyoruz
tdata=pd.DataFrame(columns=['path','class','label'])
tdata['path']=tpaths
tdata['class']=tclasses
tdata['label']=tdata['class'].map(normal_mapping)

transform = transforms.Compose([
        transforms.RandomRotation(10),  # 10 dereceye kadar döndür
        transforms.RandomHorizontalFlip(),  # yatay olarak çevir
        transforms.Resize(224),  # 224x224 boyutuna göre yeniden boyutlandır
        transforms.CenterCrop(224),  # merkezinden 224x224 alan kırp
        transforms.ToTensor(),  # tensöre çevir
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # normalizasyon
])

# Görüntü yolu ve etiket listesi fonksiyonu
def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows(): # DataFrame'in her satırını dolaştırır
        path = row['path']
        label = row['label']
        path_label_list.append((path, label))  # listeye çiftleri ekler
    return path_label_list

# Eğitim verisi için yol ve etiketleri oluşturur
path_label = create_path_label_list(data)
# Veriyi karıştırır
path_label = random.sample(path_label, len(path_label))
# Eğitim verisinin uzunluğunu ve ilk 3 örneği ekrana yazdırır
print(len(path_label))
print(path_label[0:3])

# Test verisi için yol ve etiketleri oluştur
tpath_label = create_path_label_list(tdata)
tpath_label = random.sample(tpath_label, len(tpath_label))
print(len(tpath_label))
print(tpath_label[0:3])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]  # belirtilen indeks için görüntü yolu ve etiketi al
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)  # Dönüşümleri uygula

        return img, label

class ImageDataset(pl.LightningDataModule):
    def __init__(self, path_label, batch_size=32):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'Test':
            dataset = CustomDataset(self.path_label, self.transform)
            dataset_size = len(dataset)
            print(dataset_size)
            self.test_dataset = torch.utils.data.Subset(dataset, range(dataset_size))  # Test kümesini belirle
        else:
            dataset = CustomDataset(self.path_label, self.transform)  # Eğitim/Doğrulama veri kümesi oluştur
            dataset_size = len(dataset)
            train_size = int(0.8 * dataset_size)  # %80 eğitim, %20 doğrulama olarak böl
            val_size = dataset_size - train_size
            print(train_size, val_size)
            self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))  # Eğitim kümesi
            self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))  # Doğrulama kümesi

    def __len__(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        elif self.val_dataset is not None:
            return len(self.val_dataset)
        else:
            return 0

    def __getitem__(self, index):
        if self.train_dataset is not None:
            return self.train_dataset[index]
        elif self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise IndexError("Index out of range. The dataset is empty.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  # Eğitim veri yükleyici

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)  # Doğrulama veri yükleyici

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)  # Test veri yükleyici

class DataModule(pl.LightningDataModule):
    def __init__(self, transform=None, batch_size=16):
        super().__init__()
        self.root_dir = "dir0"  # Eğitim dizini
        self.test_dir = "dir2"  # Test dizini
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'Test':
            test_set = datasets.ImageFolder(root=self.test_dir, transform=self.transform)
            self.test_dataset = DataLoader(test_set, batch_size=self.batch_size)
        else:
            dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
            n_data = len(dataset)
            n_train = int(0.8 * n_data)  # %80 eğitim
            n_val = n_data - n_train  # %20 doğrulama
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
            self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # Eğitim veri yükleyici
            self.val_dataset = DataLoader(val_dataset, batch_size=self.batch_size)  # Doğrulama veri yükleyici

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset

# Konvolüsyonel Sinir Ağı (CNN) modelini tanımlayan sınıf
class ConvolutionalNetwork(LightningModule):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        # İlk konvolüsyonel katman (3 giriş kanalı, 6 çıkış kanalı, 3x3 çekirdek)
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        # İkinci konvolüsyonel katman (6 giriş, 16 çıkış, 3x3 çekirdek)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Tam bağlı katmanlar
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))  # Son katman, sınıf sayısına bağlı

    def forward(self, X):
        X = F.relu(self.conv1(X))  # İlk konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))  # İkinci konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)  # Düzleştirme
        X = F.relu(self.fc1(X))  # İlk tam bağlı katman + ReLU
        X = F.relu(self.fc2(X))  # İkinci " " " " "
        X = F.relu(self.fc3(X))  # Üçüncü " " " " "
        X = self.fc4(X)  # Çıkış katmanı
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)  # Model tahminlerini al
        loss = F.cross_entropy(y_hat, y)  # Kayıp fonksiyonunu hesapla
        pred = y_hat.argmax(dim=1, keepdim=True)  # En yüksek olasılıklı tahmini al
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]  # Doğruluk hesapla
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)

dataset = ImageDataset(path_label)
dataset.setup()
train_dataloader = dataset.train_dataloader  # Eğitim veri yükleyicisi
val_dataloader = dataset.val_dataloader  # Doğrulama veri yükleyicisi

datamodule = DataModule()
datamodule.setup()
model = ConvolutionalNetwork()

trainer = pl.Trainer(max_epochs=10)  # 10 epoch boyunca eğitilecek
trainer.fit(model, datamodule)  # Modeli eğit

# Doğrulama veri yükleyicisi ile test et
val_loader = datamodule.val_dataloader()
trainer.test(dataloaders=val_loader)

# İlk batch'i al ve görselleştir
for images, labels in datamodule.val_dataloader():
    break
im = make_grid(images, nrow=8)

plt.figure(figsize=(12,12))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))

# Normalizasyonu geri alma
inv_normalize = transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
                                     std=[1/0.229,1/0.224,1/0.225])
im = inv_normalize(im)

plt.figure(figsize=(12,12))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))

# Modeli CPU'ya taşı ve değerlendirme moduna al
device = torch.device("cpu")
model.eval()
y_true = []
y_pred = []

# Doğrulama seti üzerinden tahmin yap
with torch.no_grad():
    for test_data in datamodule.val_dataloader():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Eğitim istatistiklerini kaydet
for epoch in range(trainer.max_epochs):
    if "train_loss" in trainer.callback_metrics:
        train_losses.append(trainer.callback_metrics["train_loss"].item())
    if "val_loss" in trainer.callback_metrics:
        val_losses.append(trainer.callback_metrics["val_loss"].item())
    if "train_acc" in trainer.callback_metrics:
        train_accs.append(trainer.callback_metrics["train_acc"].item())
    if "val_acc" in trainer.callback_metrics:
        val_accs.append(trainer.callback_metrics["val_acc"].item())

# Modeli test moduna al
device = torch.device("cpu")
model.eval()

y_true = []
y_pred = []

# Modelin tahminlerini al
with torch.no_grad():
    for test_data in datamodule.val_dataloader():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        outputs = model(test_images)
        predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()  # En yüksek olasılıklı sınıfı al
        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(predicted_classes)

# Numpy dizisine dönüştür
y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
f1_scores = [report[class_name]['f1-score'] for class_name in class_names]

plt.figure(figsize=(10, 5))
sns.barplot(x=class_names, y=f1_scores, palette="viridis")
plt.xlabel("Classes")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Class")
plt.xticks(rotation=45)
plt.ylim(0, 1)  # F1-score 0 ile 1 arasında olacağı için sınırları belirledik
plt.show()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
