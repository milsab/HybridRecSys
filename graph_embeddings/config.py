config_dict = {
  "experiment_name": "GRAPH_EMBEDDING",
  "dataset_name": "KR",
  "dataset_sample_ration": 1,

  "epochs": 1,
  "learning_rate": 0.01,

  "input_size": 64,
  "hidden_size": 128,
  "embedding_size": 64,
  "batch_size": 64,

  "feedforward_network_hidden_size": 32,  # use for creating original embedding out of the node features
  "feedforward_network_learning_rate": 0.001,  # use for creating original embedding out of the node features

  "timeframe": "W",
  "temporal": False,  # if 'True' then we add timestamp as edge features
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

  "n_clusters": 5
}


class Config:
    def __init__(self):

        self.experiment_name = config_dict.get('experiment_name')
        self.dataset_name = config_dict.get('dataset_name')
        self.dataset_sample_ration = config_dict.get('dataset_sample_ration')

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