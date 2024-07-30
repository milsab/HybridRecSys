import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GAE


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(16, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels=embedding_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Feedforward Neural Network to Create embeddings with higher dimensions
class EmbeddingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ------------------ Graph Autoencoder ------------------ #
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, dropout):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim, cached=True)  # cached only for transductive learning
        self.gcn2 = GCNConv(hidden_dim, out_channels, cached=True)  # cached only for transductive learning
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, head, dropout):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=head, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * head, out_channels, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout to node features
        x = self.gat2(x, edge_index)
        return x


class GraphAutoEncoder(GAE):
    def __init__(self, encoder):
        super(GraphAutoEncoder, self).__init__(encoder)


