import numpy as np
import pandas as pd
import math
import torch

from torch_geometric.data import Data


# Load dataset
def load_dataset(path, sample_ratio):

    df = pd.read_csv(path)
    df = df[['user_id', 'item_id', 'timestamp']]
    df = df.sample(frac=sample_ratio)  # sampling the dataset

    return df


def load_train_test(path, sample_ratio, sort=False, n=1):
    """
    Split data into training and test sets based on the last n interactions per user.
    :param sort: if true, we sort the dataset based on timestamp first and then split accordingly
    :param path: path to the dataset file
    :param sample_ratio:
    :param n: number of hold out interactions per user
    :return:
    """
    df = pd.read_csv(path)
    df = df[['user_id', 'item_id', 'rating', 'timestamp']]
    df = df.sample(frac=sample_ratio)  # sampling the dataset

    # Map user IDs to unique integer indices starts from zero
    user_ids = pd.factorize(df.user_id)[0]
    df.user_id = user_ids
    # Map item IDs to unique integer indices starts from the last user_id
    item_ids = pd.factorize(df.item_id)[0] + len(set(user_ids))
    df.item_id = item_ids

    # Convert timestamps to UNIX time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].astype('int64') // 1e9

    # Normalize the UNIX timestamps to use as edge attributes
    df['edge_attr'] = (df['timestamp'] - df['timestamp'].min()) / (
            df['timestamp'].max() - df['timestamp'].min())

    if sort:
        df = df.sort_values(by=['user_id', 'timestamp'])

    test_idx = df.groupby('user_id').tail(n).index
    train_df = df.drop(test_idx)
    test_df = df.loc[test_idx]

    return train_df, test_df


def get_users_items(df):
    # Map user IDs to unique integer indices starts from zero
    user_ids = pd.factorize(df.user_id)[0]
    # Map item IDs to unique integer indices starts from the last user_id
    item_ids = pd.factorize(df.item_id)[0] + len(set(user_ids))

    return user_ids, item_ids


def create_bipartite_graph_timestamp(df, user_ids, item_ids):
    # Create the edge index tensor
    edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)

    # Create edge weight tensor
    edge_attr = torch.tensor(df['edge_weight'].values, dtype=torch.float)

    # Create a simple bipartite graph dataset
    bi_graph = Data(edge_index=edge_index, edge_attr=edge_attr)

    return bi_graph


def get_data(dataset_path, dataset_sample_ration):
    df = load_dataset(dataset_path, dataset_sample_ration)
    user_ids, item_ids = get_users_items(df)
    bi_graph = create_bipartite_graph(user_ids, item_ids)

    return df, user_ids, item_ids, bi_graph


def get_data_timestamp(dataset_path, dataset_sample_ration):
    df = load_dataset(dataset_path, dataset_sample_ration)
    # Convert timestamps to UNIX time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['unix_timestamp'] = df['timestamp'].astype('int64') // 1e9

    # Normalize the UNIX timestamps to use as edge weights, if necessary
    # It's often a good idea to scale these to a [0, 1] range or standardize them
    df['edge_weight'] = (df['unix_timestamp'] - df['unix_timestamp'].min()) / (
                df['unix_timestamp'].max() - df['unix_timestamp'].min())

    user_ids, item_ids = get_users_items(df)

    bi_graph = create_bipartite_graph_timestamp(df, user_ids, item_ids)

    return df, user_ids, item_ids, bi_graph


def get_data_timestamp_2(dataset_path, dataset_sample_ration):
    train_df, test_df = load_train_test(dataset_path, dataset_sample_ration)

    # Convert timestamps to UNIX time
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    train_df['unix_timestamp'] = train_df['timestamp'].astype('int64') // 1e9

    # Normalize the UNIX timestamps to use as edge weights, if necessary
    # It's often a good idea to scale these to a [0, 1] range or standardize them
    train_df['edge_weight'] = (train_df['unix_timestamp'] - train_df['unix_timestamp'].min()) / (
            train_df['unix_timestamp'].max() - train_df['unix_timestamp'].min())

    user_ids, item_ids = get_users_items(train_df)

    bi_graph = create_bipartite_graph_timestamp(train_df, user_ids, item_ids)

    return train_df, test_df, user_ids, item_ids, bi_graph


def split_data(path, sample_ratio, test_ratio, split_manner, convert_to_timestamp=False):
    df = load_dataset(path, sample_ratio)

    # Exclude users with only one interaction
    counts = df['user_id'].value_counts()
    valid_users = counts[counts > 1].index
    df = df[df['user_id'].isin(valid_users)]

    # Map user IDs to unique integer indices starts from zero
    new_user_ids = pd.factorize(df.user_id)[0]
    df.user_id = new_user_ids
    # Map item IDs to unique integer indices starts from the last user_id
    item_ids = pd.factorize(df.item_id)[0] + len(set(new_user_ids))
    df.item_id = item_ids

    # Convert timestamps to UNIX time
    if convert_to_timestamp:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].astype('int64') // 1e9

    # Normalize the UNIX timestamps to use as edge attributes
    df['edge_attr'] = (df['timestamp'] - df['timestamp'].min()) / (
            df['timestamp'].max() - df['timestamp'].min())

    # Function to split data for a single user with random selection
    def split_user_data_random(user_df):
        n_items = len(user_df)
        n_test = math.ceil(n_items * test_ratio)
        test_indices = np.random.choice(user_df.index, size=n_test, replace=False)
        train_indices = list(set(user_df.index) - set(test_indices))
        return user_df.loc[train_indices], user_df.loc[test_indices]

    # Function to split data for a single user with temporal manner
    def split_user_data_temporal(user_df):
        user_df = user_df.sort_values('timestamp')
        n_items = len(user_df)
        n_test = math.ceil(n_items * test_ratio)
        n_train = n_items - n_test
        train_user = user_df.head(n_train)
        test_user = user_df.tail(n_test)
        return train_user, test_user

    # Group by user and apply splitting function
    train_list = []
    test_list = []
    for user_id, group in df.groupby('user_id'):
        if split_manner == 'random':
            train_user, test_user = split_user_data_random(group)
        elif split_manner == 'temporal':
            train_user, test_user = split_user_data_temporal(group)
        else:
            raise '"split_manner" not recognized.'
        train_list.append(train_user)
        test_list.append(test_user)

    # Concatenate all individual user splits into train and test DataFrames
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df


def create_bipartite_graph(df, temporal):
    # Ensure user_ids and item_ids are consecutive and non-overlapping
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()

    # Mapping user_ids and item_ids to graph node indices
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_mapping = {item_id: idx + len(unique_users) for idx, item_id in enumerate(unique_items)}

    # Creating edge index from user_id and item_id
    edge_index = torch.tensor([
        [user_mapping[row['user_id']], item_mapping[row['item_id']]]
        for _, row in df.iterrows()
    ]).t().contiguous()

    # Include edge attributes like timestamp
    if temporal:
        edge_attr = torch.tensor(df['edge_attr'].values, dtype=torch.float).reshape(-1, 1)
        bi_graph = Data(x=None, edge_index=edge_index, edge_attr=edge_attr)
    else:
        bi_graph = Data(x=None, edge_index=edge_index)

    return bi_graph
