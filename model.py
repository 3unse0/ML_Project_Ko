import torch.nn as nn

# Define input and output size for the neural network
input_size = 28 * 28
output_size = 10

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear(out)
        return out
