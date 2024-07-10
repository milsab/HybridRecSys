import numpy as np
import time
import mlflow
import dagshub
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

# ----------------------------------------- Load .env file -----------------------------------------
exp_num, machine_name, wandb_key, dataset_path, mlflow_tracking_uri, dagshub_owner, dagshub_repo = utils.read_env()

exp_num = 's' + str(
    exp_num) if machine_name == 'Server-GPU' else exp_num  # add "s" in the run-name if it runs on Server

# ----------------------------------------- Hyperparameters -----------------------------------------
EPOCHS = 30
EMBEDDING_SIZE = 64
INPUT_SIZE = 64  # Number of features for each Node (Users & Items)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64  # Embedding Size
N_CLUSTERS = 5  # Number of Clusters for K-MEAN
DATASET_SAMPLE_RATIO = 0.001
ATTENTION_HEAD = 4
DROPOUT = 0
EXPERIMENT_NAME = 'GRAPH_EMBEDDINGS'

DATASET_NAME = 'GR'
DATASET_PATH = dataset_path
DATASET_FILE = f'{DATASET_PATH}kairec_big_core5.csv' if DATASET_NAME == 'KR' else f'{DATASET_PATH}goodreads_core50.csv'
CONVERT_TO_TIMESTAMP = True if DATASET_NAME == 'GR' else False

TEMPORAL = True  # if 'True' it means that we add timestamp as edge features
SPLIT_MANNER = 'random'  # if 'temporal' then split is based on temporal, if 'random' then it is based on random split

RUN_NAME = f'{exp_num}_{DATASET_NAME}_GAT_TEMPORAL_{EMBEDDING_SIZE}_spl_{SPLIT_MANNER}' if TEMPORAL \
    else f'{exp_num}_{DATASET_NAME}_GAT_{EMBEDDING_SIZE}_spl_{SPLIT_MANNER}'

print(RUN_NAME)

# -------------------------------- Load Data & Create Bipartite Graph --------------------------------
train_df, test_df = preprocessing.split_data(DATASET_FILE, DATASET_SAMPLE_RATIO, split_manner=SPLIT_MANNER,
                                             test_ratio=0.2, convert_to_timestamp=CONVERT_TO_TIMESTAMP)
bi_graph = preprocessing.create_bipartite_graph(train_df, temporal=TEMPORAL)


# -------------------------------- MLFlow, DagsHub, and WANDB Setups --------------------------------

# Set DagsHub & MLFlow
dagshub.init(dagshub_repo, dagshub_owner, mlflow=True)
mlflow.set_tracking_uri(mlflow_tracking_uri)
# mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.start_run(run_name=RUN_NAME)

mlflow.set_tag('type', f'{RUN_NAME}')
mlflow.set_tag('DS File', DATASET_FILE)
mlflow.set_tag('Data Size', train_df.shape[0])
mlflow.set_tag('No. of Users', train_df.user_id.nunique())
mlflow.set_tag('No. of Items', train_df.item_id.nunique())

mlflow.log_param('EPOCHS', EPOCHS)
mlflow.log_param('EMBEDDING_SIZE', EMBEDDING_SIZE)
mlflow.log_param('INPUT_SIZE', INPUT_SIZE)
mlflow.log_param('HIDDEN_SIZE', HIDDEN_SIZE)
mlflow.log_param('OUTPUT_SIZE', OUTPUT_SIZE)
# mlflow.log_param('K-KMeans', KMeans)
mlflow.log_param('DATASET_SAMPLE_RATIO', DATASET_SAMPLE_RATIO)
mlflow.log_param('SPLIT', SPLIT_MANNER)

# Set WANDB
hyper_params = dict(
    DS_file=DATASET_FILE,
    DS_size=train_df.shape[0],
    users=train_df.user_id.nunique(),
    items=train_df.item_id.nunique(),
    embedding_size=EMBEDDING_SIZE,
    in_dim=INPUT_SIZE,
    hid_dim=HIDDEN_SIZE,
    out_dim=OUTPUT_SIZE,
    att_head=ATTENTION_HEAD,
    machine=machine_name,
    clusters=N_CLUSTERS,
    model='GAE',
    epochs=EPOCHS,
    split=SPLIT_MANNER
    # model=model.__class__.__name__
)

utils.set_wandb(wandb_key, RUN_NAME, hyper_params)


# ----------------------------------------- Run the Model -----------------------------------------
user_indices = np.arange(start=0, stop=len(set(train_df.user_id)), step=1)
item_indices = np.arange(start=len(set(train_df.user_id)), stop=len(set(train_df.user_id)) + len(set(train_df.item_id)),
                         step=1)

model, embeddings = experiment.run_graph_autoencoder(bi_graph, EMBEDDING_SIZE, OUTPUT_SIZE, HIDDEN_SIZE,
                                                     train_df.user_id.nunique(), train_df.item_id.nunique(),
                                                     user_indices, item_indices,
                                                     head=ATTENTION_HEAD, dropout=DROPOUT, epochs=EPOCHS)

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

wandb.log({'Precision': precision, 'Recall': recall,
           'NDCG': ndcg, 'Hit-Ratio': hit_ratio,
           'data_size': train_df.shape[0]})

# ----------------------------------------- Finishing -----------------------------------------
utils.update_env()  # update experiment_number in the env file


execution_time = time.time() - start_time
mlflow.log_metric('Execution-Time', execution_time)
wandb.log({'Execution_Time': execution_time})

mlflow.end_run()
wandb.finish()

print(f"Execution time for {RUN_NAME}: {execution_time} seconds")
