import h5py
import wandb
import json
import pandas as pd


def save_embeddings(embeddings, path, dataset_name):
    with h5py.File(path, 'w') as f:
        # Create dataset with compression
        f.create_dataset(dataset_name, data=embeddings, compression='gzip', compression_opts=9)


def load_embeddings(path, dataset_name):
    with h5py.File(path, 'r') as f:
        # Load the data back into memory
        embeddings = f[dataset_name][:]
        return embeddings


def set_wandb(wandb_key, wandb_name, hyper_params):
    wandb.login(key=wandb_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project="GraphEmbedding",
        name=wandb_name,

        # track hyper-parameters and run metadata
        config=hyper_params
    )


def read_env():
    with open('env_graph_embedding.json') as file:
        env_dict = json.load(file)
        exp_num = env_dict['EXPERIMENT_No']
        machine = env_dict['MACHINE']
        wandb_key = env_dict['WANDB_KEY']
        dataset_path = env_dict['DATASET_PATH']
    return exp_num, machine, wandb_key, dataset_path


def update_env():
    with open('env_graph_embedding.json', 'r') as file:
        env_dict = json.load(file)
        env_dict['EXPERIMENT_No'] = env_dict['EXPERIMENT_No'] + 1

    with open('env_graph_embedding.json', 'w') as file:
        json.dump(env_dict, file, indent=4)


def aux_load():
    df = pd.read_csv('datasets/data_started.csv')
    df = df[['user_id', 'book_id', 'rating', 'started_at']]
    df.rename(columns={'book_id': 'item_id', 'started_at': 'timestamp'}, inplace=True)
    df.to_csv('datasets/goodreads_full.csv', index=False)

    return df

