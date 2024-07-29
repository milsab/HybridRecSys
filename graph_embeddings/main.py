import numpy as np
import time
import mlflow
import wandb
import utils
import preprocessing
import clustering
import ranking
from run import Run
import plot
import evaluation

# Record the start time
start_time = time.time()

log = utils.get_log()
# ----------------------------------------- Load env file -----------------------------------------
log.info('Load env. File ')
exp_num, machine_name, wandb_key, dataset_path, mlflow_tracking_uri, dagshub_owner, dagshub_repo = utils.read_env()

exp_num = 's' + str(
    exp_num) if machine_name == 'Server-GPU' else exp_num  # add "s" in the run-name if it runs on Server

# ----------------------------------------- Load config file -----------------------------------------
log.info('Load Config File')
configs = utils.load_config()

RUN_NAME = f'{exp_num}_{configs.dataset_name}_{configs.experiment_type}'
print(RUN_NAME)

dataset_file = dataset_path + configs.KR_dataset_file if configs.dataset_name == 'KR' \
    else dataset_path + configs.GR_dataset_file

utils.set_seed(configs.seed)

# ----------------------------------------- Set MLFlow and WANDB -----------------------------------------
log.info('Set MLFlow & WANDB')
utils.set_mlflow(machine_name, RUN_NAME, mlflow_tracking_uri, dagshub_owner, dagshub_repo, dataset_file)
utils.set_wandb(wandb_key, RUN_NAME, machine_name, dataset_file)

# -------------------------------- Load Data --------------------------------
log.info('Load Data')

train_df, val_df, test_df = preprocessing.split_data(dataset_file,
                                                     sample_ratio=configs.dataset_sample_ratio,
                                                     split_manner=configs.split_manner,
                                                     test_ratio=configs.test_ratio,
                                                     val_ratio=configs.val_ratio,
                                                     convert_to_timestamp=configs.convert_to_timestamp)

mlflow.set_tag('Data Size', train_df.shape[0])
mlflow.set_tag('No. of Users', train_df.user_id.nunique())
mlflow.set_tag('No. of Items', train_df.item_id.nunique())

wandb.log({
    'data_size': train_df.shape[0],
    'users': train_df.user_id.nunique(),
    'items': train_df.item_id.nunique()
})

# ----------------------------------------- Run Experiment ----------------------------------------
log.info('Run Experiment')
run = Run(train_df, val_df, configs)
model, embeddings = run.start()

mlflow.set_tag('Model', model.__class__.__name__)
wandb.log({'model': model.__class__.__name__})

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
log.info('Ranking')
k = 5
user_indices = np.arange(start=0, stop=len(set(train_df.user_id)), step=1)
item_indices = np.arange(start=len(set(train_df.user_id)), stop=len(set(train_df.user_id)) + len(set(train_df.item_id)),
                         step=1)
user_embeddings = embeddings[user_indices]
item_embeddings = embeddings[item_indices]

recommendation = ranking.get_top_k(user_embeddings, item_embeddings, k=k)

# ----------------------------------------- Evaluation -----------------------------------------
log.info('Evaluation')
hit_ratio = evaluation.evaluate_hits(test_df, recommendation)
precision, recall = evaluation.precision_recall_at_k(recommendation, test_set=test_df, k=k)
ndcg = evaluation.ndcg(recommendation, test_set=test_df, k=k)

print(f'Hit Ratio: {hit_ratio:.4f}')
print(f'Precision_at_{k}: {precision:.4f}')
print(f'Recall_at_{k}: {recall:.4f}')
print(f'NDCG_at_{k}: {ndcg:.4f}')

mlflow.log_metric('Hit-Ratio on Test Data', hit_ratio)
mlflow.log_metric(f'Precision_at_{k} on Test Data', precision)
mlflow.log_metric(f'Recall_at_{k} on Test Data', recall)
mlflow.log_metric(f'NDCG_at_{k} on Test Data', ndcg)

wandb.log({
    'Precision on Test Data': precision,
    'Recall on Test Data': recall,
    'NDCG on Test Data': ndcg,
    'Hit-Ratio on Test Data': hit_ratio
})

# ----------------------------------------- Finishing -----------------------------------------
log.info('Finishing')
utils.update_env()  # update experiment_number in the env file

execution_time = time.time() - start_time
mlflow.log_metric('Execution-Time', execution_time)
wandb.log({'Execution_Time': execution_time})

mlflow.end_run()
wandb.finish()

print(f"Execution time for {RUN_NAME}: {execution_time} seconds")
