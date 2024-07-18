import os
import pandas as pd

import preprocessing
import experiment


class Run:
    def __init__(self, train_df, configs):
        self.train_df = train_df
        self.configs = configs
        self.experiments = {
            'time_edge_random': (self.__time_edge_random, ()),
            'time_edge_original': (self.__time_edge_original, ()),
            'time_snapshot_iterative': (self.__time_snapshot_iterative, ())
        }

    def start(self):
        exp, args = self.experiments.get(self.configs.experiment_type, (self.__unknown_exp, ()))
        return exp(*args)

    def __time_edge_random(self):
        print(f"============= Experiment Type: __time_edge_random =============")
        bi_graph = preprocessing.create_bipartite_graph(self.train_df,
                                                        temporal=True)  # temporal=true adds time as edge_att

        # embeddings=0 will initialize embeddings (node features: Data.x) randomly
        model, embeddings = experiment.run_graph_autoencoder(bi_graph, initial_embeddings='random',
                                                             num_users=self.train_df.user_id.nunique(),
                                                             num_items=self.train_df.item_id.nunique()
                                                             )
        return model, embeddings

    def __time_edge_original(self):
        print(f"============= Experiment Type: __time_edge_original =============")
        bi_graph = preprocessing.create_bipartite_graph(self.train_df,
                                                        temporal=True)  # temporal=true adds time as edge_att

        # embeddings=0 will initialize embeddings (node features: Data.x) with original users and items features
        model, embeddings = experiment.run_graph_autoencoder(bi_graph, initial_embeddings=None,
                                                             num_users=self.train_df.user_id.nunique(),
                                                             num_items=self.train_df.item_id.nunique()
                                                             )
        return model, embeddings
        return

    def __time_snapshot_iterative(self):
        print(f"============= Experiment Type: time_snapshot_iterative =============")
        snapshot_files = self.__get_snapshots()

        embeddings = None
        for file_name in snapshot_files:
            filepath = os.path.join(self.configs.snapshots_dir, file_name)
            print(f" --------------- Processing {filepath} ---------------")

            snapshot_df = pd.read_csv(filepath)
            snapshot_bi_graph = preprocessing.create_bipartite_graph(snapshot_df, temporal=False)

            model, embeddings = experiment.run_graph_autoencoder(snapshot_bi_graph,
                                                                 initial_embeddings=embeddings,
                                                                 num_users=self.train_df.user_id.nunique(),
                                                                 num_items=self.train_df.item_id.nunique(),
                                                                 )
            embeddings = embeddings.detach()  # detach from the computation graph

        return model, embeddings

    def __time_snapshots_rnn(self):
        return

    def __time_snapshots_transformers(self):
        return

    def __unknown_exp(self):
        print('The experiment is unknown!')
        model = None
        embeddings = None

        return model, embeddings

    def __get_snapshots(self):
        snapshots_dir = self.configs.snapshots_dir

        # Use a regular expression to extract the numeric part of the filename and converts it to an integer
        import re

        def numeric_key(filename):
            return int(re.search(r'(\d+)', filename).group())

        # Sorts the filenames based on the numeric part extracted by the numeric_key function.
        file_list = sorted(os.listdir(snapshots_dir), key=numeric_key)
        return file_list