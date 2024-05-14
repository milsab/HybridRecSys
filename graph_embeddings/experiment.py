import torch


# Initialize Embedding Randomly
def initialize_embedding_random(item_ids, embedding_dim):
    x = torch.randn((max(item_ids) + 1, embedding_dim), requires_grad=True)
    return x


def initialize_embedding_timestamp():
    return 0