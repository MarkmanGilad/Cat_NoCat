import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        # -> n, 3, 64, 64
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 30, 30
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 13, 13
        x = x.view(-1, 16 * 13 * 13)            # -> n, 2704
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        x = torch.sigmoid(x)
        return x