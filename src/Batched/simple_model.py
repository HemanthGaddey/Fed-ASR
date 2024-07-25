import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1=nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten(  )
        self.fc1 = nn.Linear(64*14*14*4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x