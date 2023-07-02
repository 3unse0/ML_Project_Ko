#  Digit Recognition using MNIST Dataset

This program trains a neural network to classify handwritten digits using the MNIST dataset. It utilizes the PyTorch library for deep learning.
<br>
<br>
<br>
## Prerequisites

Before running the program, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- matplotlib
<br>

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies
4. To run the program, execute the 'main.py' script
<br>

## Program Structure

The program consists of the following files:

- 'main.py': The main script that sets up the data loaders, neural network model, loss function, optimizer, and performs the training and testing.
- 'datasets.py': Contains custom dataset classes for loading the MNIST training and test datasets.
- 'model.py': Defines the neural network model architecture.
- 'utils.py': Provides utility functions for training and testing the model.
<br>

## Usage

The program will train the neural network model on the MNIST training dataset and evaluate its performance on the test dataset. After training, it will randomly select an example from the test dataset and make predictions on it.

The accuracy and loss of the model will be displayed during training, and the predicted and actual labels for the random example will be printed to the console. Additionally, an image of the example will be displayed using matplotlib.
<br>
<br>
<br>
<img width="1035" alt="output" src="https://github.com/3unse0/ML_Project_Ko/assets/130636819/9f49d905-31e1-47df-a290-76fa054eef50">

