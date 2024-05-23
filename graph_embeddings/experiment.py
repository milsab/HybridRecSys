import torch
import wandb
from torch_geometric.nn import GATConv, GAE

import models
import ranking
import evaluation
import mlflow


# Initialize Embedding Randomly
def initialize_embedding_random(item_ids, embedding_dim):
    x = torch.randn((max(item_ids) + 1, embedding_dim), requires_grad=True)
    return x


def initialize_embedding_timestamp():
    return 0


# Training Loop
def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_graph_autoencoder(bi_graph, in_channels, out_channel, hidden_size, num_users, num_items,
                          user_indices, item_indices, head, dropout, epochs):
    # Parameters
    in_channels = in_channels  # number of features per node
    out_channels = out_channel  # size of the latent space

    num_user_features = in_channels
    num_item_features = in_channels

    user_features = torch.randn(num_users, num_user_features)
    item_features = torch.randn(num_items, num_item_features)

    # Concatenate user and item features into a single node feature matrix
    node_features = torch.cat([user_features, item_features], dim=0)
    bi_graph.x = node_features

    # Model Initialization
    encoder = models.GATEncoder(in_channels, out_channels, hidden_size, head, dropout)
    model = models.GraphAutoencoder(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = bi_graph.to(device)
    model = model.to(device)

    for epoch in range(epochs):
        loss = train(data, model, optimizer)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        wandb.log({'Train Loss': loss})
        # embeddings_after_each_epoch = model.encode(data.x, data.edge_index)
        # get_evaluation(data, embeddings_after_each_epoch, user_indices, item_indices)


    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)

    return model, embeddings


def get_evaluation(data, embeddings, user_indices, item_indices, k=5):

    embeddings = embeddings.cpu().detach().numpy()
    user_embeddings = embeddings[user_indices]
    item_embeddings = embeddings[item_indices]

    recommendations = ranking.get_top_k(user_embeddings, item_embeddings, k=k)

    hit_ratio = evaluation.evaluate_hits(data, recommendations)
    precision, recall = evaluation.precision_recall_at_k(recommendations, test_set=data, k=k)
    ndcg = evaluation.ndcg(recommendations, test_set=data, k=k)

    wandb.log({'Train Hit-Ratio': hit_ratio})
    wandb.log({'Train Precision': hit_ratio, 'Train Recall': recall})
    wandb.log({'Train NDCG': ndcg})

