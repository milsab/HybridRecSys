import h5py
import wandb
import json
import mlflow
import dagshub
from config import Config
import pandas as pd


def load_config():
    return Config()


def read_env():
    with open('env_graph_embedding.json') as file:
        env_dict = json.load(file)
        exp_num = env_dict['EXPERIMENT_No']
        machine = env_dict['MACHINE']
        wandb_key = env_dict['WANDB_KEY']
        dataset_path = env_dict['DATASET_PATH']
        mlflow_tracking_uri = env_dict['MLFLOW_TRACKING_URI']
        dagshub_owner = env_dict['DAGSHUB_OWNER']
        dagshub_repo = env_dict['DAGSHUB_REPO_NAME']
    return exp_num, machine, wandb_key, dataset_path, mlflow_tracking_uri, dagshub_owner, dagshub_repo


def update_env():
    with open('env_graph_embedding.json', 'r') as file:
        env_dict = json.load(file)
        env_dict['EXPERIMENT_No'] = env_dict['EXPERIMENT_No'] + 1

    with open('env_graph_embedding.json', 'w') as file:
        json.dump(env_dict, file, indent=4)


def save_embeddings(embeddings, path, dataset_name):
    with h5py.File(path, 'w') as f:
        # Create dataset with compression
        f.create_dataset(dataset_name, data=embeddings, compression='gzip', compression_opts=9)


def load_embeddings(path, dataset_name):
    with h5py.File(path, 'r') as f:
        # Load the data back into memory
        embeddings = f[dataset_name][:]
        return embeddings


def set_wandb(wandb_key, wandb_name, dataset_file, machine_name):
    config = load_config()
    wandb.login(key=wandb_key)
    hyper_params = dict(
        DS_file=dataset_file,
        embedding_size=config.embedding_size,
        in_dim=config.input_size,
        hid_dim=config.hidden_size,
        att_head=config.model['attention_head'],
        machine=machine_name,
        clusters=config.n_clusters,
        model_approach=config.model['type'],
        epochs=config.epochs,
        split=config.split_manner,
        timeframe=config.timeframe
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project="GraphEmbedding",
        name=wandb_name,

        # track hyper-parameters and run metadata
        config=hyper_params
    )


def set_mlflow(machine_name, run_name, mlflow_tracking_uri, dagshub_owner, dagshub_repo, dataset_file):
    config = load_config()
    # Set DagsHub & MLFlow
    dagshub.init(dagshub_repo, dagshub_owner, mlflow=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # mlflow.set_tracking_uri('http://localhost:5000/')

    mlflow.set_experiment(config.experiment_name)
    mlflow.start_run(run_name=run_name)

    mlflow.set_tag('type', f'{run_name}')
    mlflow.set_tag('DS File', dataset_file)

    mlflow.log_param('EPOCHS', config.epochs)
    mlflow.log_param('EMBEDDING_SIZE', config.embedding_size)
    mlflow.log_param('INPUT_SIZE', config.input_size)
    mlflow.log_param('HIDDEN_SIZE', config.hidden_size)

    # mlflow.log_param('K-KMeans', KMeans)
    mlflow.log_param('DATASET_SAMPLE_RATIO', config.dataset_sample_ration)
    mlflow.log_param('SPLIT', config.split_manner)
    mlflow.log_param('TIMEFRAME', config.timeframe)


def get_sparsity(df):
    num_of_users = df.user_id.nunique()
    num_of_items = df.item_id.nunique()

    total_elements_in_matrix = num_of_users * num_of_items
    num_of_non_zero_elements = df.shape[0]
    num_of_zero_elements = total_elements_in_matrix - num_of_non_zero_elements

    sparsity = num_of_zero_elements / total_elements_in_matrix

    return sparsity * 100

