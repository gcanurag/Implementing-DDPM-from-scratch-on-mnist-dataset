import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load the MNIST dataset
transform = transforms.ToTensor()  # keeps pixel values as [0,1]
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Get one sample image and its label
image, label = mnist[1]  # image shape: (1, 28, 28)

# Remove the channel dimension for plotting
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()
