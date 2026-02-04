# CIFAR-10 Image Classification using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset into 10 categories.

Dataset:
- CIFAR-10 (60,000 RGB images, 32×32)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded automatically using torchvision

Model:
Input → Conv + ReLU + MaxPool → Conv + ReLU + MaxPool → Fully Connected → Output

Training:
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Batch size: 64
- Epochs: 5

Results:
- Test accuracy: ~69.9%
- Visualization of predictions with correct (green) and incorrect (red) labels

Run:
python cnn_cifar10.py

This project demonstrates practical CNN implementation, training, evaluation, and visualization using PyTorch.
