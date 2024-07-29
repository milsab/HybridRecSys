import numpy as np
import pandas as pd
import torch


def get_top_k(user_embeddings, item_embeddings, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(user_embeddings, np.ndarray):
        user_embeddings = torch.from_numpy(user_embeddings).to(device)
    if isinstance(item_embeddings, np.ndarray):
        item_embeddings = torch.from_numpy(item_embeddings).to(device)

    # Compute dot product
    dot_product = torch.matmul(user_embeddings, item_embeddings.T)

    # top_values contains the top k scores
    # top_indices contains the indices of the top k items
    # top_scores, top_indices = torch.topk(dot_product, k, dim=1, largest=True, sorted=True)

    top_k = torch.topk(dot_product, k, dim=1).indices

    # Convert to a dictionary of user_id to item_id list
    recommendations = {user: items.tolist() for user, items in enumerate(top_k)}

    return recommendations


def get_ground_truth(df):
    # Assuming df is our original DataFrame with columns ['user_id', 'item_id', 'timestamp', 'rating']
    positive_threshold = 4  # Define what rating counts as a positive interaction
    filtered_df = df[df['rating'] >= positive_threshold]

    # Group by user_id and aggregate item_ids into a set
    ground_truth = filtered_df.groupby('user_id')['item_id'].apply(set).to_dict()

    return ground_truth

