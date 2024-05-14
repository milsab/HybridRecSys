import numpy as np
import time
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


# Load env file
exp_num, machine_name, wandb_key, _ = utils.read_env()
exp_num = 's' + str(exp_num) if machine_name == 'Server-GPU' else exp_num # add "s" in the run-name if it runs on Server

# Hyperparameters
EMBEDDING_SIZE = 32
INPUT_SIZE = 32  # Number of features for each Node (Users & Items)
HIDDEN_SIZE = 64
OUTPUT_SIZE = 32  # Embedding Size
N_CLUSTERS = 5  # Number of Clusters for K-MEAN
DATASET_PATH = 'datasets/comic_data.csv'
DATASET_SAMPLE_RATIO = 0.1
ATTENTION_HEAD = 1
DROPOUT = 0
EXPERIMENT_NAME = 'GRAPH_EMBEDDINGS'
TEMPORAL = False
RUN_NAME = f'{exp_num}_GAT_TEMPORAL_{EMBEDDING_SIZE}' if TEMPORAL else f'{exp_num}_GAT_{EMBEDDING_SIZE}'

start_time = time.time()  # Record the start time

mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.start_run(run_name=RUN_NAME)


# Get data
train_df, test_df = preprocessing.split_data(DATASET_PATH, DATASET_SAMPLE_RATIO, 0.2)
bi_graph = preprocessing.create_bipartite_graph(train_df, temporal=TEMPORAL)

# Node2Vec
# user_embeddings, item_embeddings = models.node_2_vec(bi_graph, train_df.user_id.unique(), train_df.item_id.unique())
# model = 'N2V'

# Initialize the model
# model = models.GCN(hidden_channels=HIDDEN_SIZE, embedding_dim=EMBEDDING_SIZE)
model = models.GAT(num_features=EMBEDDING_SIZE, hidden_channels=HIDDEN_SIZE, out_channels=OUTPUT_SIZE,
                   heads=ATTENTION_HEAD, dropout=DROPOUT)

# Initialize embeddings randomly
random_embedding = experiment.initialize_embedding_random(train_df.item_id, EMBEDDING_SIZE)

# Forward pass to get new embeddings
embeddings = model(random_embedding, bi_graph.edge_index)
print(embeddings)

# Save embeddings as NumPy using h5py with compression
utils.save_embeddings(embeddings.detach().cpu().numpy(), f'../Embedding/{RUN_NAME}.h5', 'embeddings')


hyper_params = dict(
            data_size=train_df.shape[0],
            embedding_size=EMBEDDING_SIZE,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            machine=machine_name,
            num_clusters=N_CLUSTERS,
            model=model.__class__.__name__
        )

utils.set_wandb(wandb_key, RUN_NAME, hyper_params)

# Load embeddings
embeddings = utils.load_embeddings(f'../Embedding/{RUN_NAME}.h5', 'embeddings')

# # get Clusters
# labels = clustering.get_kmeans_clusters(embeddings, N_CLUSTERS)
#
# # Visualizing
# plot.plot_PCA(embeddings, kmeans_labels=labels, num_users=len(set(train_df.user_id)), num_clusters=N_CLUSTERS,
#               run_name=RUN_NAME)

# Ranking
k = 5
user_indices = np.arange(start=0, stop=len(set(train_df.user_id)), step=1)
item_indices = np.arange(start=len(set(train_df.user_id)), stop=len(set(train_df.user_id)) + len(set(train_df.item_id)),
                         step=1)

user_embeddings = embeddings[user_indices]
item_embeddings = embeddings[item_indices]

recommendation = ranking.get_top_k(user_embeddings, item_embeddings, k=k)

# Evaluating
hit_ratio = evaluation.evaluate_hits(test_df, recommendation)
print(f'Hit Ratio: {hit_ratio:.4f}')
mlflow.log_metric('Hit-Ratio', hit_ratio)

# Precision, Recall
precision, recall = evaluation.precision_recall_at_k(recommendation, test_set=test_df, k=k)
print(f'Precision_at_{k}: {precision:.4f}')
print(f'Recall_at_{k}: {recall:.4f}')
mlflow.log_metric(f'Precision_at_{k}', precision)
mlflow.log_metric(f'Recall_at_{k}', recall)

# NDCG
ndcg = evaluation.ndcg(recommendation, test_set=test_df, k=k)
print(f'NDCG_at_{k}: {ndcg:.4f}')
mlflow.log_metric(f'NDCG_at_{k}', ndcg)

mlflow.log_artifact(f'../Embedding/{RUN_NAME}.h5')
mlflow.set_tag('type', f'{RUN_NAME}')
mlflow.set_tag('data_size', train_df.shape[0])
mlflow.set_tag('Model', model.__class__.__name__)
mlflow.log_param('EMBEDDING_SIZE', EMBEDDING_SIZE)
mlflow.log_param('INPUT_SIZE', INPUT_SIZE)
mlflow.log_param('HIDDEN_SIZE', HIDDEN_SIZE)
mlflow.log_param('OUTPUT_SIZE', OUTPUT_SIZE)
mlflow.log_param('K-KMeans', KMeans)
mlflow.log_param('DATASET_SAMPLE_RATIO', DATASET_SAMPLE_RATIO)


end_time = time.time()  # Record the end time
execution_time = end_time - start_time

mlflow.log_metric('Execution-Time', execution_time)
mlflow.end_run()

wandb.log({'Precision': precision, 'Recall': recall,
           'NDCG': ndcg, 'Hit-Ratio': hit_ratio,
           'data_size': train_df.shape[0],
           'Execution_Time': execution_time})
wandb.finish()
utils.update_env() # update experiment_number in the env file

print(f"Execution time for {RUN_NAME}: {execution_time} seconds")
