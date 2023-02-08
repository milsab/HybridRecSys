import torch
import torch.nn as nn
import torch.nn.functional as F


class RecSys(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecSys, self).__init__()
        self.linear1 = nn.Linear(input_size, 384)
        self.linear2 = nn.Linear(input_size, 192)
        self.linear3 = nn.Linear(192, output_size)

    def forward(self, x):
        x = x.view(-1, 768)
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        # # out = self.linear3(out)
        # out = torch.sigmoid(self.linear3(out))
        # # out = F.softmax(self.linear3(out), dim=0)
        return out


# Child class to deal with binary classification in case we set targets like(1) or dislike(0)
class RecSysBinary(RecSys):
    def __init__(self, n_input_size, output_size):
        super(RecSysBinary, self).__init__(n_input_size, output_size)

    def forward(self, x):
        out = super().forward(x)
        return torch.sigmoid(self.linear3(out))


# Child class to deal with multi classification in case we set targets as rates from 1 to 5
class RecSysMulti(RecSys):
    def __init__(self, n_input_size, output_size):
        super(RecSysMulti, self).__init__(n_input_size, output_size)

    def forward(self, x):
        out = super().forward(x)
        return torch.softmax(self.linear3(out), dim=0)
