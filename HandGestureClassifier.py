import torch
import torch.nn as nn
import torch.nn.functional as F


class HandGestureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HandGestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = F.softmax(self.fc3(x))
        return out
