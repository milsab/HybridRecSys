import torch
import torch.nn as nn
import torch.nn.functional as F


class RecSys(nn.Module):
    def __init__(self, input_size, output_size, dropout=False):
        super(RecSys, self).__init__()
        self.linear1 = nn.Linear(input_size, 384)
        self.linear2 = nn.Linear(384, 192)
        self.linear3 = nn.Linear(192, output_size)

        self.dropout = dropout      # adding dropout technique for regularization

    def forward(self, x):
        x = x.view(-1, 768)
        out = F.relu(self.linear1(x))
        if self.dropout:
            out = F.dropout(out)
        out = F.relu(self.linear2(out))
        return out


# Child class to deal with binary classification in case we set targets like(1) or dislike(0)
class RecSysBinary(RecSys):
    def __init__(self, n_input_size, output_size, dropout=False):
        super(RecSysBinary, self).__init__(n_input_size, output_size, dropout=False)

    def forward(self, x):
        out = super().forward(x)
        return torch.sigmoid(self.linear3(out))


# Child class to deal with multi classification in case we set targets as rates from 1 to 5
class RecSysMulti(RecSys):
    def __init__(self, n_input_size, output_size, dropout=False):
        super(RecSysMulti, self).__init__(n_input_size, output_size, dropout=False)

    def forward(self, x):
        out = super().forward(x)
        # return self.linear3(out)
        return torch.softmax(self.linear3(out), dim=1)


class BaselineCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors, bias=False):
        super(BaselineCF, self).__init__()

        self.user_factors = nn.Embedding(n_users, n_factors)
        if bias:
            self.user_bias = nn.Embedding(n_users, 1)

        self.item_factors = nn.Embedding(n_items, n_factors)
        if bias:
            self.item_bias = nn.Embedding(n_items, 1)

        self.bias = bias

    def forward(self, data):
        users = self.user_factors(data[:, 0])  # matrix with user_ids by user_factors
        items = self.item_factors(data[:, 1])  # matrix with item_ids by item_factors

        out = (users * items).sum(dim=1, keepdim=True)
        if self.bias:
            out += self.user_bias(data[:, 0]) + self.item_bias(data[:, 1])

        return torch.sigmoid(out)


