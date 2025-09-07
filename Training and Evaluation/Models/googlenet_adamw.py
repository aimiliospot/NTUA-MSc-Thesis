import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision
import pandas as pd
import os
from tqdm import tqdm
from evaluateFunction import evaluateGoogleNet
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import MulticlassConfusionMatrix
from torch import Tensor

# Create necessary directories if they do not exist
directories = [
    "figures/",
    "conf_matrices/",
    "models/"
]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

model_name = 'googlenet_adamw'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load train_df used to map images to classes
train_dataframe = pd.read_csv('train_dataframe.csv')

transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
    ])
train_dataset = datasets.ImageFolder(
    root='./trainDataset/',
    transform = transformation,
)

# Dictionary to map images to classes
class_map = {}
for index, row in train_dataframe.iterrows():
    class_map[row["image_title"]] = row["class_label"]

# Split the dataset into training and testing sets
train_size = int(0.7 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_set, test_set = random_split(train_dataset, [train_size, test_size])


# Custom dataset class to load the images and class labels
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_map):
        self.dataset = dataset
        self.class_map = class_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_index = self.dataset.indices[index]
        image, _ = self.dataset.dataset[original_index]
        image_title = os.path.basename(self.dataset.dataset.imgs[original_index][0])  # If your dataset has 'imgs' attribute
        class_label = self.class_map[image_title]

        return image, class_label

# Train and Test Dataloaders
train_dataloader = DataLoader(TrainDataset(train_set, class_map), batch_size=64, shuffle=True)
test_dataloader = DataLoader(TrainDataset(test_set, class_map), batch_size=64, shuffle=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# Model initialization
model = torchvision.models.GoogLeNet()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience= 5)

# Variables needed for the training process
test_accuracy_per_epoch = []
test_loss_per_epoch = []
train_loss_per_epoch = []
min_val_loss = 10000
epochs_that_min_val_loss_does_not_reduce = 0

for epoch in range(100):
    print('Epoch: ',epoch + 1)
    sum_of_losses_per_epoch = 0
    batches_number = 0
    for images, labels in tqdm(train_dataloader):
        batches_number += 1
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images).logits
        
        # Training loss calculation
        loss = criterion(outputs, labels)
        sum_of_losses_per_epoch += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum_of_losses_per_epoch/batches_number
    train_loss_per_epoch.append(avg_loss)
    print('Training Loss: ', avg_loss)

    # Calculation of validation loss and validation acccuracy
    test_accuracy, test_loss = evaluateGoogleNet(model, test_dataloader, device, criterion)
    test_accuracy_per_epoch.append(test_accuracy)
    test_loss_per_epoch.append(test_loss)
    print("Testing accuracy:", test_accuracy)
    print("Testing loss:", test_loss)
    scheduler.step(test_loss)
    print(f"Epoch {epoch}, Learning Rate {optimizer.param_groups[0]['lr']:.6f}")

    # Criterion to stop training
    if test_loss < min_val_loss:
        min_val_loss = test_loss
        epochs_that_min_val_loss_does_not_reduce = 0
    else:
        epochs_that_min_val_loss_does_not_reduce += 1

    print('Epochs that val loss didn\'t reduce:', epochs_that_min_val_loss_does_not_reduce)
    
    if epochs_that_min_val_loss_does_not_reduce == 15: 
        print("Stopping early as validation loss didn't reduce for 10 consecutive epochs.")
        break

# Save model
torch.save(model.state_dict(), f'./models/{model_name}.pth')

# Confusion matrix initialization
final_confusion_matrix = torch.zeros((4, 4))

# Calculation of confusion matrix 
for images, labels in tqdm(test_dataloader):
    images = images.to(device)
    labels = labels.to(device)
        
    # Forward pass
    with torch.no_grad():
        outputs = model(images).logits
        metric = MulticlassConfusionMatrix(4)
        metric.update(outputs, labels)
        confusion_matrix = metric.compute()
        final_confusion_matrix += confusion_matrix

# Save confusion matrix
torch.save(final_confusion_matrix, f'./conf_matrices/conf_matrix_{model_name}.pth')

# Generation of Test Accuracy, Test Loss and Train Loss plots
plt.figure(figsize=(8, 6))
plt.plot(test_accuracy_per_epoch, marker='o', linestyle='-')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.savefig(f'./figures/test_accuracy_{model_name}.png')

plt.figure(figsize=(8, 6))
plt.plot(test_loss_per_epoch, marker='o', linestyle='-')
plt.title('Test Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.grid(True)
plt.savefig(f'./figures/test_loss_per_epoch_{model_name}.png')

plt.figure(figsize=(8, 6))
plt.plot(train_loss_per_epoch, marker='o', linestyle='-')
plt.title('Train Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.grid(True)
plt.savefig(f'./figures/train_loss_per_epoch_{model_name}.png')


