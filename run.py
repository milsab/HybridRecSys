import shutil
import time
import wandb


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import preprocessing
import data
import experiment

# Default Hyper-Parameters
DATASET_PATH = '../../MyExperiments/datasets/goodreads/comics/'
TENSORBOARD_PATH = '../tensorboard/'
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001
TRAINING_RATIO = 0.9
TEST_RATIO = 0.1
WEIGHT_DECAY = 0
DROPOUT = False


class Run:
    def __init__(self, model, criterion, optimizer,
                 ratings_filename, users_embeddings_filename, items_embeddings_filename, tensorboard_name, wandb_name,
                 select_data_size,
                 ds_path=DATASET_PATH,
                 batch_size=BATCH_SIZE,
                 training_ratio=TRAINING_RATIO, test_ratio=TEST_RATIO,
                 epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 lr_scheduler=None, dropout=DROPOUT):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.ratings_filename = ratings_filename
        self.users_embeddings_filename = users_embeddings_filename
        self.items_embeddings_filename = items_embeddings_filename
        self.tensorboard_name = tensorboard_name
        self.wandb_name = wandb_name
        self.data_size = select_data_size

        self.ds_path = ds_path

        self.batch_size = batch_size

        self.training_ratio = training_ratio
        self.test_ratio = test_ratio

        self.learning_rate = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.dropout = dropout

    def __reset_tensorboard(self):
        shutil.rmtree(TENSORBOARD_PATH + self.tensorboard_name, ignore_errors=True)
        # Wait for 1 second in order to tensorboard related folder get deleted correctly
        time.sleep(3)

    def __set_wandb_ai(self):
        hyper_params = dict(
            epochs=self.epochs,
            data_size=self.data_size,
            init_lr=self.learning_rate,
            tr_ratio=self.training_ratio,
            ts_ratio=self.test_ratio,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            dropout=self.dropout,
            optimizer=self.optimizer.__class__.__name__,
            loss=self.criterion.__class__.__name__,
            lr_schedule=self.lr_scheduler.__class__.__name__,
            model=self.model.__class__.__name__
        )

        wandb.init(
            # set the wandb project where this run will be logged
            project="HybridRecSys_BERT",
            name=self.wandb_name,
            # track hyper-parameters and run metadata
            config=hyper_params
        )

    def start(self):

        # Set WANDB.AI
        if self.wandb_name:
            self.__set_wandb_ai()

        self.__reset_tensorboard()
        print('DEVICE IS: ', self.device)

        # Load Ratings
        ratings = preprocessing.load_data(path=self.ds_path, filename=self.ratings_filename)
        ratings = ratings[:self.data_size]

        # Split Data
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(ratings, self.training_ratio,
                                                                                  self.test_ratio)

        # Load Embeddings
        users_embeddings = preprocessing.load_data(path=self.ds_path, filename=self.users_embeddings_filename)
        items_embeddings = preprocessing.load_data(path=self.ds_path, filename=self.items_embeddings_filename)

        # Create DataLoaders
        train_ds = data.GoodreadsDataset(x_train, y_train, users_embeddings, items_embeddings)
        train_loader = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True)

        val_ds = data.GoodreadsDataset(x_val, y_val, users_embeddings, items_embeddings)
        val_loader = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=True)

        test_ds = data.GoodreadsDataset(x_test, y_test, users_embeddings, items_embeddings)
        test_loader = DataLoader(dataset=test_ds, batch_size=self.batch_size)

        # Set Criterion
        criterion = self.criterion

        # Set Optimizer
        optimizer = self.optimizer

        # Set Tensorboard
        writer = SummaryWriter(TENSORBOARD_PATH + self.tensorboard_name)

        # Execute Experiment
        my_experiment = experiment.Experiment(self.model, train_loader, val_loader, test_loader,
                                              criterion, optimizer, self.device, writer, self.epochs,
                                              self.learning_rate)

        my_experiment.run(self.lr_scheduler)
