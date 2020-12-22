import torch.nn as nn
import torch.nn.functional as F


class HNN(nn.Module):
    def __init__(self):
      super(HNN, self).__init__()
      self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, 5, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
      self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 5),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

      self.fc1 = nn.Linear(in_features=48, out_features=300)
      self.fc2 = nn.Linear(in_features=300, out_features=300)
      self.out = nn.Linear(in_features=300, out_features=10)

    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.log_softmax(self.out(x), dim=0)

      return x
