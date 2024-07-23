config_dict = {
  "experiment_name": "GRAPH_EMBEDDING",

  # "experiment_type": "no_time",   # time as edge_attr | node_features initialize randomly
  # "experiment_type": "time_edge_random",   # time as edge_attr | node_features initialize randomly
  # "experiment_type": "time_edge_original",  # time as edge_attr | node_features initialize with user&item features
  "experiment_type": "time_snapshot_iterative_random",  # time as series of snapshots with iterative approach (random node features)
  # "experiment_type": "time_snapshot_iterative_original",  # time as series of snapshots with iterative approach


  "dataset_name": "KR",
  "KR_dataset_file": "kairec_big_core5.csv",
  "GR_dataset_file": "goodreads_core50.csv",
  "dataset_sample_ratio": 1,
  "test_ratio": 0.2,
  "snapshots_dir": "datasets/snapshots/KR",

  "regenerate_bi_graph": False,  # if False => will load bi_graph from file. If True => regenerate bi_graph

  "epochs": 30,
  "learning_rate": 0.001,

  "input_size": 64,
  "hidden_size": 128,
  "embedding_size": 64,
  "batch_size": 64,

  "feedforward_network_hidden_size": 128,  # use for creating original embedding out of the node features
  "feedforward_network_learning_rate": 0.01,  # use for creating original embedding out of the node features

  "timeframe": "W",
  "temporal": True,  # if 'True' then we add timestamp as edge features
  "split_manner": "temporal",  # if 'temporal' => Temporal Split, if 'random' => Random Split

  "model": {
    "type": "GAE",
    "attention_head": 4,
    "layers": 5,
    "dropout": 0.3
  },

  "optimizer": {
    "type": "Adam",
    "momentum": 0.9
  },

  "n_clusters": 5,
  "torch_seed": 7

}


class Config:
    def __init__(self):

        self.experiment_name = config_dict.get('experiment_name')
        self.experiment_type = config_dict.get('experiment_type')

        self.dataset_name = config_dict.get('dataset_name')
        self.KR_dataset_file = config_dict.get('KR_dataset_file')
        self.GR_dataset_file = config_dict.get('GR_dataset_file')

        self.dataset_sample_ratio = config_dict.get('dataset_sample_ratio')
        self.test_ratio = config_dict.get('test_ratio')
        self.snapshots_dir = config_dict.get('snapshots_dir')

        self.epochs = config_dict.get('epochs')
        self.learning_rate = config_dict.get('learning_rate')

        self.input_size = config_dict.get('input_size')
        self.hidden_size = config_dict.get('hidden_size')
        self.embedding_size = config_dict.get('embedding_size')
        self.batch_size = config_dict.get('batch_size')

        self.feedforward_network_hidden_size = config_dict.get('feedforward_network_hidden_size')
        self.feedforward_network_learning_rate = config_dict.get('feedforward_network_learning_rate')

        self.model = config_dict.get('model')
        self.optimizer = config_dict.get('optimizer')

        self.timeframe = config_dict.get('timeframe')

        self.temporal = config_dict.get('temporal')

        self.split_manner = config_dict.get('split_manner')

        self.n_clusters = config_dict.get('n_clusters')

        self.torch_seed = config_dict.get('torch_seed')

        self.regenerate_bi_graph = config_dict.get('regenerate_bi_graph')

        self.convert_to_timestamp = True if self.dataset_name == 'GR' else False
