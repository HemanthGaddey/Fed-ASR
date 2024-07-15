import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 10)
        self.model=resnet18

    def forward(self, x):
        x = self.model(x)
        return x