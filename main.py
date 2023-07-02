import torch
import random
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from datasets import MNISTTrainDataset, MNISTTestDataset
from model import NeuralNet
from utils import train, test
import matplotlib.pyplot as plt

# Set up batch size and data loaders
batch_size = 100
train_dataset = MNISTTrainDataset('data', transform=ToTensor())
test_dataset = MNISTTestDataset('data', transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create an instance of the neural network model and move it to the device
model = NeuralNet().to(device)

# Set up the loss function, optimizer, and learning rate
learning_rate = 0.001
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Set the number of epochs for training
epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n -------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")

# Load the real test dataset for a random example
real_test_dataset = MNISTTestDataset('data')
rand = random.randint(0, 9999)
X, y = test_dataset[rand][0], test_dataset[rand][1]
A = real_test_dataset[rand][0]

# Make predictions on the random example
with torch.no_grad():
    pred = model(X.unsqueeze(0).to(device))
    predicted, actual = pred[0].argmax(0), y
    print(f"predicted:", predicted, ", actual:", actual)
    plt.imshow(A, cmap='gray')
    plt.show()
