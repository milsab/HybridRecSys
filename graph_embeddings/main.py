import numpy as np
import pandas as pd
import time
import os
import mlflow
import wandb
import utils
import models
import preprocessing
import clustering
import experiment
import ranking
import plot
import evaluation
from sklearn.cluster import KMeans

# Record the start time
start_time = time.time()

# ----------------------------------------- Load env file -----------------------------------------
exp_num, machine_name, wandb_key, dataset_path, mlflow_tracking_uri, dagshub_owner, dagshub_repo = utils.read_env()

exp_num = 's' + str(
    exp_num) if machine_name == 'Server-GPU' else exp_num  # add "s" in the run-name if it runs on Server


# ----------------------------------------- Load config file -----------------------------------------
configs = utils.load_config()

DATASET_FILE = f'{dataset_path}kairec_big_core5.csv' if configs.dataset_name == 'KR' else f'{dataset_path}goodreads_core50.csv'
CONVERT_TO_TIMESTAMP = True if configs.dataset_name == 'GR' else False

RUN_NAME = f'{exp_num}_{configs.dataset_name}_GAT_TEMPORAL_{configs.embedding_size}_spl_{configs.split_manner}' if configs.temporal \
    else f'{exp_num}_{configs.dataset_name}_GAT_{configs.embedding_size}_spl_{configs.split_manner}'
print(RUN_NAME)

# ----------------------------------------- Set MLFlow and WANDB -----------------------------------------
utils.set_mlflow(machine_name, RUN_NAME, mlflow_tracking_uri, dagshub_owner, dagshub_repo, DATASET_FILE)
utils.set_wandb(wandb_key, RUN_NAME, DATASET_FILE, machine_name)

# -------------------------------- Load Data & Create Bipartite Graph --------------------------------
train_df, test_df = preprocessing.split_data(DATASET_FILE, configs.dataset_sample_ration, split_manner=configs.split_manner,
                                             test_ratio=0.2, convert_to_timestamp=CONVERT_TO_TIMESTAMP)

mlflow.set_tag('Data Size', train_df.shape[0])
mlflow.set_tag('No. of Users', train_df.user_id.nunique())
mlflow.set_tag('No. of Items', train_df.item_id.nunique())

# ----------------------------------------- Run the Model -----------------------------------------

snapshots_dir = 'datasets/snapshots/KR'
embeddings = None

import re


# Use a regular expression to extract the numeric part of the filename and converts it to an integer
def numeric_key(filename):
    return int(re.search(r'(\d+)', filename).group())


# Sorts the filenames based on the numeric part extracted by the numeric_key function.
file_list = sorted(os.listdir(snapshots_dir), key=numeric_key)

for filename in file_list:
    filepath = os.path.join(snapshots_dir, filename)
    print(f"============= Processing {filepath} =============")
    snapshot_df = pd.read_csv(filepath)
    snapshot_bi_graph = preprocessing.create_bipartite_graph(snapshot_df, temporal=False)

    model, embeddings = experiment.run_graph_autoencoder(snapshot_bi_graph, embeddings,
                                                         configs.input_size, configs.embedding_size, configs.hidden_size,
                                                         snapshot_df.user_id.nunique(), snapshot_df.item_id.nunique(),
                                                         head=configs.model['attention_head'],
                                                         dropout=configs.model['dropout'], epochs=configs.epochs)

    embeddings = embeddings.detach()  # detach from the computation graph


# bi_graph = preprocessing.create_bipartite_graph(train_df, temporal=TEMPORAL)
# model, embeddings = experiment.run_graph_autoencoder(bi_graph, EMBEDDING_SIZE, OUTPUT_SIZE, HIDDEN_SIZE,
#                                                      train_df.user_id.nunique(), train_df.item_id.nunique(),
#                                                      user_indices, item_indices,
#                                                      head=ATTENTION_HEAD, dropout=DROPOUT, epochs=EPOCHS)

mlflow.set_tag('Model', model.__class__.__name__)

# Save embeddings as NumPy using h5py with compression
utils.save_embeddings(embeddings.detach().cpu().numpy(), f'../Embedding/{RUN_NAME}.h5', 'embeddings')
mlflow.log_artifact(f'../Embedding/{RUN_NAME}.h5')

# Load embeddings
embeddings = utils.load_embeddings(f'../Embedding/{RUN_NAME}.h5', 'embeddings')

# ----------------------------------------- Clustering -----------------------------------------
# labels = clustering.get_kmeans_clusters(embeddings, N_CLUSTERS)

# ----------------------------------------- Visualization --------------------------------------
# plot.plot_PCA(embeddings, kmeans_labels=labels, num_users=len(set(train_df.user_id)), num_clusters=N_CLUSTERS,
#               run_name=RUN_NAME)

# ----------------------------------------- Ranking --------------------------------------------
k = 5
user_indices = np.arange(start=0, stop=len(set(train_df.user_id)), step=1)
item_indices = np.arange(start=len(set(train_df.user_id)), stop=len(set(train_df.user_id)) + len(set(train_df.item_id)),
                         step=1)
user_embeddings = embeddings[user_indices]
item_embeddings = embeddings[item_indices]

recommendation = ranking.get_top_k(user_embeddings, item_embeddings, k=k)

# ----------------------------------------- Evaluation -----------------------------------------
hit_ratio = evaluation.evaluate_hits(test_df, recommendation)
precision, recall = evaluation.precision_recall_at_k(recommendation, test_set=test_df, k=k)
ndcg = evaluation.ndcg(recommendation, test_set=test_df, k=k)

print(f'Hit Ratio: {hit_ratio:.4f}')
print(f'Precision_at_{k}: {precision:.4f}')
print(f'Recall_at_{k}: {recall:.4f}')
print(f'NDCG_at_{k}: {ndcg:.4f}')

mlflow.log_metric('Hit-Ratio', hit_ratio)
mlflow.log_metric(f'Precision_at_{k}', precision)
mlflow.log_metric(f'Recall_at_{k}', recall)
mlflow.log_metric(f'NDCG_at_{k}', ndcg)

wandb.log({
    'Precision': precision, 'Recall': recall,
    'NDCG': ndcg, 'Hit-Ratio': hit_ratio,
    'data_size': train_df.shape[0],
    'users': train_df.user_id.nunique(),
    'items': train_df.item_id.nunique()
})

# ----------------------------------------- Finishing -----------------------------------------
utils.update_env()  # update experiment_number in the env file

execution_time = time.time() - start_time
mlflow.log_metric('Execution-Time', execution_time)
wandb.log({'Execution_Time': execution_time})

mlflow.end_run()
wandb.finish()

print(f"Execution time for {RUN_NAME}: {execution_time} seconds")
