import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.utils import to_networkx
# from node2vec import Node2Vec
import networkx as nx

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


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, heads, dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):

        if edge_attr:
            # First graph attention layer with ELU activation function
            x = F.elu(self.conv1(x, edge_index, edge_attr))

            # Second graph attention layer for extracting embeddings
            x = self.conv2(x, edge_index, edge_attr)
        else:
            # First graph attention layer with ELU activation function
            x = F.elu(self.conv1(x, edge_index))

            # Second graph attention layer for extracting embeddings
            x = self.conv2(x, edge_index)

        return x


class GAT_v2(torch.nn.Module):
    def __init__(self, user_feature_size, item_feature_size, hidden_size, output_size, heads):
        super(GAT, self).__init__()
        self.user_gat = GATConv(user_feature_size, hidden_size, heads=heads, concat=True)
        self.item_gat = GATConv(item_feature_size, hidden_size, heads=heads, concat=True)
        self.fc = torch.nn.Linear(hidden_size * heads, output_size)

    def forward(self, data):
        user_x, item_x, edge_index = data.user_x, data.item_x, data.edge_index
        user_x = self.user_gat(user_x, edge_index)
        user_x = torch.nn.functional.elu(user_x)
        item_x = self.item_gat(item_x, edge_index)
        item_x = torch.nn.functional.elu(item_x)
        x = torch.cat([user_x, item_x], dim=0)
        x = self.fc(x)
        return x


# def node_2_vec(data, user_ids, item_ids):
#     # Convert PyG data object to NetworkX Graph to ba able to run Node2Vec
#     G = to_networkx(data, to_undirected=True)
#
#     node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
#     model = node2vec.fit(window=10, min_count=1, batch_words=4)
#
#     # Get embeddings for users and items
#     user_embeddings = {user: model.wv[user] for user in user_ids}
#     item_embeddings = {item: model.wv[item] for item in item_ids}
#
#     return user_embeddings, item_embeddings
