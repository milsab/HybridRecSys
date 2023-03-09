import shutil
import time
import wandb
import csv
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import preprocessing
import data
import experiment

# Default Hyper-Parameters
# DATASET_PATH = '../../MyExperiments/datasets/goodreads/comics/'
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
                 # ds_path=DATASET_PATH,
                 batch_size=BATCH_SIZE,
                 training_ratio=TRAINING_RATIO, test_ratio=TEST_RATIO,
                 epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 lr_scheduler=None, dropout=DROPOUT):

        # Read 'env' file to set environment parameters
        machine, wandb_key, dataset_path = self.__read_env()

        self.machine_name = machine
        self.wandb_key = wandb_key
        self.ds_path = dataset_path

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

        # self.ds_path = ds_path

        self.batch_size = batch_size

        self.training_ratio = training_ratio
        self.test_ratio = test_ratio

        self.learning_rate = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.dropout = dropout

        self.hyper_params = dict(
            epochs=self.epochs,
            data_size=self.data_size,
            init_lr=self.learning_rate,
            tr_ratio=self.training_ratio,
            ts_ratio=self.test_ratio,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            dropout=self.dropout,
            device=self.device,
            machine=self.machine_name,
            optimizer=self.optimizer.__class__.__name__,
            loss=self.criterion.__class__.__name__,
            lr_schedule=self.lr_scheduler.__class__.__name__,
            model=self.model.__class__.__name__
        )

    def __read_env(self):
        with open('../env.json') as file:
            env_dict = json.load(file)
            machine = env_dict['MACHINE']
            wandb_key = env_dict['WANDB_KEY']
            dataset_path = env_dict['DATASET_PATH']
        return machine, wandb_key, dataset_path

    def __reset_tensorboard(self):
        shutil.rmtree(TENSORBOARD_PATH + self.tensorboard_name, ignore_errors=True)
        # Wait for 1 second in order to tensorboard related folder get deleted correctly
        time.sleep(3)

    def __set_wandb_ai(self):
        hp = self.hyper_params
        wandb.login(key=self.wandb_key)
        wandb.init(
            # set the wandb project where this run will be logged
            project="HybridRecSys_BERT",
            name=self.wandb_name,

            # track hyper-parameters and run metadata
            config=hp
        )

    def __write_log(self, test_acc, train_loss, train_acc, val_loss, val_acc, runtime, precision, recall, f1, sk_acc):
        hp = self.hyper_params.copy()
        hp.update([('TestAcc', test_acc), ('Trn_Acc', train_acc), ('Trn_loss', train_loss),
                   ('val_Acc', val_acc), ('val_loss', val_loss), ('Runtime', runtime),
                   ('experiment_type', self.wandb_name), ('date', datetime.now()),
                   ('precision', precision), ('recall', recall), ('f1', f1), ('sk_acc', sk_acc)])
        header_names = hp.keys()
        with open('log.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header_names)
            # writer.writeheader()  # This line of code is necessary for the first time that the file is not exist
            writer.writerow(hp)

    def start(self):

        # Set WANDB.AI
        if self.wandb_name:
            self.__set_wandb_ai()

        self.__reset_tensorboard()
        print('DEVICE IS: ', self.device)

        # Load Ratings
        ratings = preprocessing.load_data(path=self.ds_path, filename=self.ratings_filename)
        ratings = ratings[:self.data_size]

        # Load Embeddings
        users_embeddings = preprocessing.load_data(path=self.ds_path, filename=self.users_embeddings_filename)
        items_embeddings = preprocessing.load_data(path=self.ds_path, filename=self.items_embeddings_filename)

        # Filter rating to only have users and items that we have embedding for them
        ratings = preprocessing.filter_data(ratings, users_embeddings, items_embeddings)

        # Split Data
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.split_data(ratings, self.training_ratio,
                                                                                  self.test_ratio)

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

        test_acc, train_loss, train_acc, val_loss, val_acc, runtime, precision, recall, f1, sk_acc = my_experiment.run(self.lr_scheduler)

        self.__write_log(test_acc, train_loss, train_acc, val_loss, val_acc, runtime, precision, recall, f1, sk_acc)
