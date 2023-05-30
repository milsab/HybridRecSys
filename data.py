import numpy as np
import torch
from torch.utils.data import Dataset


def concat_embeddings(interactions, users_embeddings, items_embeddings, data_name):
    input_embeddings = []

    for row in interactions:
        user_id = row[0]
        item_id = row[1]

        # Check whether we have user embedding for the current user_id
        if user_id not in users_embeddings:
            continue
        user_embedding = users_embeddings[user_id]

        if data_name == 'GDR':
            item_embedding = items_embeddings[item_id]['description']
        if data_name == 'AMZ':
            item_embedding = items_embeddings[item_id]['review']

        embedding = np.concatenate((user_embedding, item_embedding), axis=None)

        input_embeddings.append(embedding)

    return np.array(input_embeddings)


class GoodreadsDataset(Dataset):

    def __init__(self, inputs, targets, users_embeddings, items_embeddings, data_name):
        inputs = inputs.to_numpy()
        targets = targets.to_numpy()

        embeddings = concat_embeddings(inputs, users_embeddings, items_embeddings, data_name)

        self.x = torch.from_numpy(embeddings.astype(np.float32))
        self.x.requires_grad_(True)

        self.y = torch.from_numpy(targets.astype(np.float32))
        self.y.requires_grad_(True)

        self.n_samples = inputs.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

