import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.Layer1 = nn.Sequential(
            nn.Conv2d(1, 24, 7, stride=2),
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.Layer2 = nn.Sequential(
            nn.Conv2d(24, 48, 5, stride=1),
            nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.Layer3 = nn.Sequential(
            nn.Conv2d(48, 96, 2, stride=1, padding=1),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.FC = nn.Linear(
            96,
            3
        )

        self.Dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = x.float()
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.FC(x)
        x = F.log_softmax(x)
        return x