import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
images, labels = next(iter(train_loader))
images.shape, labels.shape

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN().to(device)
images, labels = next(iter(train_loader))
images = images.to(device)

outputs = model(images)
print(outputs.shape)
criterion  = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
      images = images.to(device)
      labels = labels.to(device)

      optimiser.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)

      loss.backward()
      optimiser.step()

      running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch: [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

def evaluate(model, loader):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted==labels).sum().item()
    return 100*correct/total
acc = evaluate(model, test_loader)
print(f"Test Accuracy: {acc:.2f}%")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt

model.eval()

images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
plt.figure(figsize=(12, 8))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    imshow(images[i])

    true_label = classes[labels[i]]
    pred_label = classes[preds[i]]

    title_color = "green" if true_label == pred_label else "red"
    plt.title(f"P: {pred_label}\nT: {true_label}", color=title_color)

plt.suptitle("CNN Predictions on CIFAR-10", fontsize=16)
plt.show()
