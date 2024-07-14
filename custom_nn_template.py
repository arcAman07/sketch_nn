import torch
import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()

    def forward(self, x):
        return x
