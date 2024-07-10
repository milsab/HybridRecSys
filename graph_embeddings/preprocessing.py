import numpy as np
import pandas as pd
import math
import torch

from torch_geometric.data import Data


def load_dataset(path, sample_ratio):

    df = pd.read_csv(path)
    df = df[['user_id', 'item_id', 'timestamp']]
    df = df.sample(frac=sample_ratio)  # sampling the dataset

    return df


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


def create_snapshots(path, sample_ratio, timeframe, filename_prefix):
    df = load_dataset(path, sample_ratio)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Group data by a specified timeframe (D: Daily, W: Weekly, M: Monthly, Y: Yearly)
    grouped = df.groupby(df['timestamp'].dt.to_period(timeframe))

    #  Create cumulative groups and save each snapshot to a CSV file
    cumulative_df = pd.DataFrame()

    for i, (period, group) in enumerate(grouped, start=1):
        cumulative_df = pd.concat([cumulative_df, group])
        filename = f'{filename_prefix}_{i}.csv'
        cumulative_df.to_csv(f'datasets/snapshots/{filename}', index=False)
        print(f"Saved {filename}")


# create_snapshots('datasets/kairec_big_core5.csv', sample_ratio=1, timeframe='W', filename_prefix='KR')