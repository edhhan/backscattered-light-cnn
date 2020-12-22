import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(in_features=35**2, out_features=300)
        #self.fc2 = nn.Linear(in_features=500, out_features=500)
        #self.fc3 = nn.Linear(in_features=500, out_features=500)
        #self.fc4 = nn.Linear(in_features=500, out_features=500)
        self.out = nn.Linear(in_features=300, out_features=5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.float()
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = F.log_softmax(self.out(x))
        return x
