import torch
import wandb
from torch_geometric.nn import GATConv, GAE

import utils
import models
import ranking
import evaluation
import preprocessing


# Initialize Embedding Randomly
def initialize_embedding_random(item_ids, embedding_dim):
    x = torch.randn((max(item_ids) + 1, embedding_dim), requires_grad=True)
    return x


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def get_original_embeddings():
    """
    This method gets original features of a node (user or item) and convert them to higher dimension using a simple
    feedforward network
    :return: embeddings with higher dimension
    """
    configs = utils.load_config()

    def get_high_dimension_embedding(original_features):
        model = models.EmbeddingNN(input_dim=original_features.shape[1],
                                   hidden_dim=configs.feedforward_network_hidden_size,
                                   output_dim=configs.embedding_size)

        embeddings = model(original_features)

        return embeddings

    original_user_features = torch.tensor(preprocessing.get_users_static_features().values, dtype=torch.float32)
    original_item_features = torch.tensor(preprocessing.get_items_static_features().values, dtype=torch.float32)

    user_feature_high_dimension = get_high_dimension_embedding(original_user_features)
    item_feature_high_dimension = get_high_dimension_embedding(original_item_features)

    high_dimension_features = torch.cat([user_feature_high_dimension, item_feature_high_dimension], dim=0)

    return high_dimension_features


def run_graph_autoencoder(bi_graph, embeddings, in_channels, out_channel, hidden_size, num_users, num_items,
                          head, dropout, epochs):
    # Parameters
    configs = utils.load_config()
    in_channels = in_channels  # number of features per node
    out_channels = out_channel  # size of the latent space

    if embeddings is None:
        node_features = get_original_embeddings()
    else:
        node_features = embeddings

    bi_graph.x = node_features

    # Model Initialization
    encoder = models.GATEncoder(in_channels, out_channels, hidden_size, head, dropout)
    model = models.GraphAutoencoder(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = bi_graph.to(device)
    model = model.to(device)

    # Training Loop
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

